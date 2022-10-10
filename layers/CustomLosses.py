# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
from torch import nn

class HingeLoss(nn.Module):
    def __init__(self,
                 margin: float=1.,
                 reduction: str='mean'):
        super(HingeLoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none']
        self.margin = margin
        self.reduction = reduction

    def forward(self, input, target):

        total_loss = self.margin * torch.relu(1. - torch.mul(input, target))
        loss = total_loss.mean(dim=-1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()

        loss = loss.mean()
        return loss
