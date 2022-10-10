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
from layers.CustomLayers import G_ResBlock

class Generator(nn.Module):
    def __init__(self,
                 noise_size: int,
                 embed_size: int,
                 in_channels: list,
                 out_channels: list,
                 block_num: int):
        super(Generator, self).__init__()
        assert in_channels.__len__() == out_channels.__len__()
        assert block_num == in_channels.__len__() and block_num >= 4

        self.block_num = block_num

        self.linear = nn.Linear(in_features=noise_size, out_features=in_channels[0]*4*4)

        self.blocks = nn.ModuleList([G_ResBlock(embed_size=noise_size+embed_size,
                                                input_size=in_channel,
                                                target_size=out_channel,
                                                kernel_size=(3, 3),
                                                padding=(1, 1))
                                     for in_channel, out_channel in zip(in_channels, out_channels)])

        self.init_conv = nn.Conv2d(in_channels=out_channels[-3], out_channels=3,
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), padding_mode='reflect')

        self.final_conv = nn.Conv2d(in_channels=out_channels[-1], out_channels=3,
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), padding_mode='reflect')

        self.init_params()

    def forward(self, noise, text_feats):
        """
        :return: small and big scale features
        """

        x = self.linear(noise)
        c_feats = torch.cat([noise, text_feats], dim=1)

        x = x.view(-1, 1024, 4, 4)

        feats = list()
        for i, block in enumerate(self.blocks):
            x = block(x, c_feats)
            if i == self.block_num - 3:
                feat = self.init_conv(x)
                feats.append(feat)

        feat = self.final_conv(x)
        # x = torch.tanh(x)
        feats.append(feat)

        return feats

    def init_params(self):

        for i, named_param in enumerate(self.named_parameters()):
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[0] != 'blocks':
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
            if name in ['linear.weight', 'linear.bias',
                        'init_conv.weight', 'init_conv.bias']:
                if name.split('.')[-1] == 'weight':
                    former_weights.append(param)
                else:
                    former_bias.append(param)
            elif name in ['final_conv.weight', 'final_conv.bias']:
                if name.split('.')[-1] == 'weight':
                    latter_weights.append(param)
                else:
                    latter_bias.append(param)
            else:
                if int(name.split('.')[1]) <= self.block_num - 3:
                    if name.split('.')[-1] == 'weight':
                        former_weights.append(param)
                    else:
                        former_bias.append(param)
                else:
                    if name.split('.')[-1] == 'weight':
                        latter_weights.append(param)
                    else:
                        latter_bias.append(param)

        return former_weights, former_bias, latter_weights, latter_bias

