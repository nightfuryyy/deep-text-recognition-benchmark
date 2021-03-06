
import os
import sys
import time
import random
import string
import argparse
from warpctc_pytorch import CTCLoss 
#from torch_baidu_ctc import CTCLoss
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from torchsummary import summary
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.tensorboard import SummaryWriter
fol_ckpt = '/content/drive/MyDrive/thesis/ckpt_all_adamw'
import os
if not os.path.exists(fol_ckpt) :
  os.mkdir(fol_ckpt)

writer = SummaryWriter(fol_ckpt)
from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model_guide import Model
from test_guide import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from SmoothCTCLoss import SmoothCTCLoss

def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '	')
    log.close()
    
    """ model configuration """
    # if 'CTC' in opt.Prediction:
    if opt.baiduCTC:
        CTC_converter = CTCLabelConverterForBaiduWarpctc(opt.character)
    else:
        CTC_converter = CTCLabelConverter(opt.character)
# else:
    Attn_converter = AttnLabelConverter(opt.character)
    opt.num_class_ctc = len(CTC_converter.character)
    opt.num_class_attn = len(Attn_converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class_ctc, opt.num_class_attn, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        # print(name)
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    print("Model:")
    print(model)
    # print(summary(model, (1, opt.imgH, opt.imgW,1)))
    """ setup loss """
    if opt.baiduCTC:
        # need to install warpctc. see our guideline.
        if opt.label_smooth :
            criterion_major_path = SmoothCTCLoss(num_classes = opt.num_class_ctc, weight = 0.05)
        else :
            criterion_major_path = CTCLoss()
        #criterion_major_path = CTCLoss(average_frames=False, reduction="mean", blank=0)
    else:
        criterion_major_path = torch.nn.CTCLoss(zero_infinity=True).to(device)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    #criterion_major_path = torch.nn.CTCLoss(zero_infinity=True).to(device)
    criterion_guide_path = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_avg_major_path = Averager()
    loss_avg_guide_path = Averager()
    # filter that only require gradient decent
    guide_parameters = []
    major_parameters = []
    guide_model_part_names = ["Transformation","FeatureExtraction","SequenceModeling_Attn","Attention"]
    major_model_part_names = ["SequenceModeling_CTC","CTC"]
    for name, param in model.named_parameters():
        if param.requires_grad :
            if name.split(".")[1] in guide_model_part_names :
                guide_parameters.append(param)
            elif name.split(".")[1] in major_model_part_names :
                major_parameters.append(param)
            # print(name)
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]
    if opt.continue_training :
      guide_parameters = []
    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer_ctc = AdamW(major_parameters, lr=opt.lr)
        if not opt.continue_training :
          optimizer_attn = AdamW(guide_parameters, lr=opt.lr)
    scheduler_ctc = get_linear_schedule_with_warmup(optimizer_ctc, num_warmup_steps=10000, num_training_steps=opt.num_iter)
    scheduler_attn = get_linear_schedule_with_warmup(optimizer_attn, num_warmup_steps=10000, num_training_steps=opt.num_iter) 
    start_iter = 0
    if opt.saved_model != '' and (not opt.continue_training):
        print(f'loading pretrained model from {opt.saved_model}')
        checkpoint = torch.load(opt.saved_model)
        start_iter = checkpoint['start_iter']+1
        if not opt.adam: 
            optimizer_ctc.load_state_dict(checkpoint['optimizer_ctc_state_dict'])
            if not opt.continue_training :
              optimizer_attn.load_state_dict(checkpoint['optimizer_attn_state_dict'])
            scheduler_ctc.load_state_dict(checkpoint['scheduler_ctc_state_dict'])
            scheduler_attn.load_state_dict(checkpoint['scheduler_attn_state_dict'])
            print(scheduler_ctc.get_lr())
            print(scheduler_attn.get_lr())
        if opt.FT:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    if opt.continue_training : 
        model.load_state_dict(torch.load(opt.saved_model))
    # print("Optimizer:")
    # print(optimizer)
    # 
    scheduler_ctc = get_linear_schedule_with_warmup(optimizer_ctc, num_warmup_steps=10000, num_training_steps=opt.num_iter,last_epoch = start_iter - 1)
    scheduler_attn = get_linear_schedule_with_warmup(optimizer_attn, num_warmup_steps=10000, num_training_steps=opt.num_iter, last_epoch = start_iter - 1)
    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------	'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}	'
        opt_log += '---------------------------------------	'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
      
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter - 1
    if opt.continue_training :
      start_iter = 0
    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        iteration += 1
        if iteration < start_iter :
            continue
        image = image_tensors.to(device)
        # print(image.size())
        text_attn, length_attn = Attn_converter.encode(labels, batch_max_length=opt.batch_max_length)
        #print("1")
        text_ctc, length_ctc = CTC_converter.encode(labels, batch_max_length=opt.batch_max_length) 
        #print("2")
        #if iteration == start_iter :
        #    writer.add_graph(model, (image, text_attn))
        batch_size = image.size(0)
        preds_major, preds_guide = model(image, text_attn[:, :-1])
        #print("10")
        preds_size = torch.IntTensor([preds_major.size(1)] * batch_size)
        if opt.baiduCTC:
            preds_major = preds_major.permute(1, 0, 2)  # to use CTCLoss format
            if opt.label_smooth :
                cost_ctc = criterion_major_path(preds_major, text_ctc, preds_size, length_ctc, batch_size)
            else :
                cost_ctc = criterion_major_path(preds_major, text_ctc, preds_size, length_ctc) / batch_size
        else:
            preds_major = preds_major.log_softmax(2).permute(1, 0, 2)
            cost_ctc = criterion_major_path(preds_major, text_ctc, preds_size, length_ctc)
        #print("3")
        # preds = model(image, text[:, :-1])  # align with Attention.forward
        target = text_attn[:, 1:]  # without [GO] Symbol
        if not opt.continue_training : 
            cost_attn = criterion_guide_path(preds_guide.view(-1, preds_guide.shape[-1]), target.contiguous().view(-1))
            optimizer_attn.zero_grad()
            cost_attn.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(guide_parameters, opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer_attn.step()
        optimizer_ctc.zero_grad()
        cost_ctc.backward()
        torch.nn.utils.clip_grad_norm_(major_parameters, opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer_ctc.step()
        scheduler_ctc.step()
        scheduler_attn.step()
        #print("4")
        loss_avg_major_path.add(cost_ctc)
        if not opt.continue_training :
            loss_avg_guide_path.add(cost_attn)
        if (iteration + 1) % 100 == 0 :
            writer.add_scalar("Loss/train_ctc", loss_avg_major_path.val(), (iteration + 1) // 100)
            loss_avg_major_path.reset()
            if not opt.continue_training :
                writer.add_scalar("Loss/train_attn", loss_avg_guide_path.val(), (iteration + 1) // 100)
                loss_avg_guide_path.reset()
        # validation part
        if (iteration + 1) % opt.valInterval == 0 : #or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion_major_path, valid_loader, CTC_converter, opt)
                model.train()
                writer.add_scalar("Loss/valid", valid_loss, (iteration + 1) // opt.valInterval)
                writer.add_scalar("Metrics/accuracy", current_accuracy, (iteration + 1) // opt.valInterval)
                writer.add_scalar("Metrics/norm_ED", current_norm_ED, (iteration + 1) // opt.valInterval)
                # loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {train_loss:0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                # loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
                # training loss and validation loss
                if not opt.continue_training :
                    loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss ctc: {loss_avg_major_path.val():0.5f}, Train loss attn: {loss_avg_guide_path.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                else :
                    loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss ctc: {loss_avg_major_path.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg_major_path.reset()
                if not opt.continue_training :
                    loss_avg_guide_path.reset()
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'{fol_ckpt}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'{fol_ckpt}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}	{current_model_log}	{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '	')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}	{head}	{dashed_line}	'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    # if 'Attn' in opt.Prediction:
                    #     gt = gt[:gt.find('[s]')]
                    #     pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}	{str(pred == gt)}	'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '	')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+3 == 0 and (not opt.continue_training):
            # print(scheduler_ctc.get_lr())
            # print(scheduler_attn.get_lr())
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_attn_state_dict': optimizer_attn.state_dict(),
            'optimizer_ctc_state_dict': optimizer_ctc.state_dict(),
            'start_iter' : iteration,
            'scheduler_ctc_state_dict' : scheduler_ctc.state_dict(),
            'scheduler_attn_state_dict': scheduler_attn.state_dict(),
            }
                , f'{fol_ckpt}/current_model.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=1000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--label_smooth', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--output_channel_GCN', type=int, default=512,
                        help='the number of output channel of GCN')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--guide_training', action='store_true', help='Whether to use guide_training (default not)')  
    parser.add_argument('--continue_training', action='store_true', help='Whether to continue training from best model')    
    
    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}-guidedTraining'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)

