import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter, Averager,CTCLabelConverterForBaiduWarpctc
from dataset import RawDataset, AlignCollate
from PIL import Image
import base64
import io
import pandas as pd
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if opt.guide_training :
      from model_guide import Model
    else :
      from model import Model
    if opt.baiduCTC:
        converter = CTCLabelConverterForBaiduWarpctc(opt.character)
    else :
        converter = CTCLabelConverter(opt.character)
    if opt.Prediction == 'Attn' :
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    opt.num_class_ctc = opt.num_class
    opt.num_class_attn = opt.num_class_ctc + 1

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device), strict = False)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    data = pd.DataFrame()
    with torch.no_grad():
        ind = 0
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                if opt.guide_training :
                    preds = model.module.inference(image, text_for_pred)
                else :
                    preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                # Select max probabilty (greedy decoding) then decode index to character
                if opt.baiduCTC:
                    if (opt.beam_search):
                      preds_index = preds
                    else :
                      _, preds_index = preds.max(2)
                      preds_index = preds_index.view(-1)
                else:
                    _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data,opt.beam_search)
            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                filename = img_name
                label = pred
                conf = round(confidence_score.item(),3)
                img = cv2.imread(filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format="PNG")
                imgStr = base64.b64encode(img_buffer.getvalue()).decode("utf-8") 

                data.loc[ind, 'img'] = '<img src="data:image/png;base64,{0:s}">'.format(imgStr)
                data.loc[ind, 'id'] = filename
                data.loc[ind, 'label'] = label
                data.loc[ind, 'conf'] = conf
                ind+=1
                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
        html_all = data.to_html(escape=False)
        if opt.is_save :
            text_file = open("result.html", "w") 
            text_file.write(html_all) 
            text_file.close() 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--output_channel_GCN', type=int, default=512,
                        help='the number of output channel of GCN')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--guide_training', action='store_true', help='Whether to use guide_training (default not)')
    parser.add_argument('--beam_search', action='store_true', help='whether to use beam_search')
    parser.add_argument('--is_save', action='store_true', help='whether to save result')
    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
