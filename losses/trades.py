"""https://github.com/yaodongyu/TRADES/blob/master/trades.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class TRADESLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits_clean, logits_adv):
        return self.criterion_kl(F.log_softmax(logits_adv, dim=1),
                                 F.softmax(logits_clean, dim=1))
