# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
from torch import nn
from layers.CustomLayers import Spatial, D_ResBlock

class Discriminator(nn.Module):
    def __init__(self,
                 combine_size: int,
                 in_channels: list,
                 out_channels: list,
                 block_num: int):
        super(Discriminator, self).__init__()
        assert in_channels.__len__() == out_channels.__len__()
        assert block_num == in_channels.__len__() and block_num >= 4

        self.block_num = block_num

        self.init_conv = nn.Conv2d(in_channels=3, out_channels=in_channels[0],
                                   kernel_size=(3, 3), stride=(1, 1),
                                   padding=(1, 1), padding_mode='reflect')

        self.final_conv = nn.Conv2d(in_channels=3, out_channels=in_channels[2],
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), padding_mode='reflect')

        self.blocks = nn.ModuleList([D_ResBlock(input_size=in_channel,
                                                target_size=out_channel,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1))
                                     for in_channel, out_channel in zip(in_channels, out_channels)])

        self.spatial = Spatial(features=out_channels[-1],
                               embed_size=combine_size)

        self.init_params()

    def forward(self, sources, c_feats):
        """
        :param sources: large and small scale sources
        :param c_feats: context features
        :return: large and small scale logits
        """
        assert sources.__len__() == 2

        former_sources, latter_sources = sources
        latter_sources = self.init_conv(latter_sources)

        feats = list()
        for i, block in enumerate(self.blocks):
            if i >= self.block_num - 4:
                if i == self.block_num - 4:
                    former_sources = self.final_conv(former_sources)
                    former_sources = block(former_sources)
                else:
                    former_sources = block(former_sources)

            latter_sources = block(latter_sources)

        former_sources = self.spatial(former_sources, c_feats)
        latter_sources = self.spatial(latter_sources, c_feats)

        former_sources = former_sources.view(-1, 1)
        latter_sources = latter_sources.view(-1, 1)

        feats.extend([former_sources, latter_sources])

        return feats

    def init_params(self):

        for named_param in self.named_parameters():

            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    stddev = math.sqrt(2 / (sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)
                else:
                    torch.nn.init.zeros_(param)

    def get_weights(self):

        former_weights, former_bias = [], []
        latter_weights, latter_bias = [], []
        for named_param in self.named_parameters():
            name, param = named_param
            if name in ['init_conv.weight', 'init_conv.bias']:
                if name.split('.')[-1] == 'weight':
                    former_weights.append(param)
                else:
                    former_bias.append(param)
            elif name in ['final_conv.weight', 'final_conv.bias',
                          'spatial.init_conv.weight', 'spatial.init_conv.bias',
                          'spatial.final_conv.weight', 'spatial.final_conv.bias']:
                if name.split('.')[-1] == 'weight':
                    latter_weights.append(param)
                else:
                    latter_bias.append(param)
            else:
                if int(name.split('.')[1]) >= self.block_num - 4:
                    if name.split('.')[-1] == 'bias':
                        latter_bias.append(param)
                    else:
                        latter_weights.append(param)
                else:
                    if name.split('.')[-1] == 'bias':
                        former_bias.append(param)
                    else:
                        former_weights.append(param)

        return former_weights, former_bias, latter_weights, latter_bias
