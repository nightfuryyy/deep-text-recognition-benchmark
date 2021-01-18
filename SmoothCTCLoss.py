
from torch import nn
from warpctc_pytorch import CTCLoss 
import torch
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SmoothCTCLoss(nn.Module):

    def __init__(self, num_classes, blank=0, weight=0.05, ):
        super(SmoothCTCLoss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.ctc = CTCLoss()
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, log_probs, targets, input_lengths, target_lengths, batch_size):
        #batch_size = list(log_probs.size())[1]
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths).to(device)

        kl_inp = log_probs.transpose(0, 1).to(device)
        kl_tar = torch.full_like(kl_inp, np.log(1. / self.num_classes))
        kldiv_loss = self.kldiv(kl_inp, kl_tar).to(device)

        #print(ctc_loss, kldiv_loss)
        loss = (1. - self.weight) * ctc_loss / batch_size + self.weight * kldiv_loss
        loss = loss.to(device)
        return loss
