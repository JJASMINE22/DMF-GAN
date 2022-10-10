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
from torch.nn import functional as F


class Attention(nn.Module):

    def __init__(self,
                 causal: bool=False,
                 use_scale: bool=True,
                 return_score: bool=False):
        super(Attention, self).__init__()

        self.causal = causal
        self.use_scale = use_scale
        self.return_score = return_score

        if use_scale:
            self.scale = nn.Parameter(data=torch.ones(size=()),
                                      requires_grad=True)

    def compute_attn(self, query, key):
        assert key.size().__len__() >= 2

        scores = torch.matmul(query,
                              key.transpose(1, 2) if key.size().__len__() == 3
                              else key.transpose(0, 1))

        if self.use_scale:
            scores *= self.scale
        # scores = torch.softmax(scores, dim=-1)

        return scores

    def get_padding_mask(self, mask):

        pad_mask = torch.unsqueeze(torch.logical_not(mask).float(), dim=-1)
        pad_mask = torch.maximum(pad_mask, pad_mask.transpose(1, 2))

        return pad_mask

    def get_sequence_mask(self, mask):

        seq_mask = torch.triu(torch.ones(size=(mask.size(1),)*2), diagonal=1)
        seq_mask = seq_mask.unsqueeze(dim=0)

        return seq_mask

    def forward(self, sources: list, mask=None):
        if mask is None:
            assert sources.__len__() == 3
        else:
            assert mask.size().__len__() == 2
            assert sources.__len__() == 2

        q = sources[0]
        k = sources[1]
        v = sources[2] if len(sources) == 3 else k

        scores = self.compute_attn(q, k)

        if mask is not None:
            attention_mask = self.get_padding_mask(mask)
            if self.causal:
                seq_mask = self.get_sequence_mask(mask)
                attention_mask = torch.maximum(attention_mask, seq_mask)
            scores -= 1e+9 * attention_mask
        scores = torch.softmax(scores, dim=-1)
        feat = torch.matmul(scores, v)

        if self.return_score:
            return feat, scores

        return feat


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 source_size: int=None,
                 embed_size: int=None,
                 multihead_num: int=None,
                 drop_rate: float=None):
        super(MultiHeadAttention, self).__init__()
        assert not embed_size % multihead_num

        self.source_size = source_size
        self.embed_size = embed_size
        self.multihead_num = multihead_num
        self.drop_rate = drop_rate

        self.linear = nn.Linear(in_features=self.embed_size, out_features=self.source_size)
        self.linear_q = nn.Linear(in_features=self.source_size, out_features=self.embed_size)
        self.linear_k = nn.Linear(in_features=self.source_size, out_features=self.embed_size)
        self.linear_v = nn.Linear(in_features=self.source_size, out_features=self.embed_size)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.source_size)

        self.init_params()

    def forward(self, inputs: list, mask=None):
        assert inputs.__len__() >= 2

        q = inputs[0]
        k = inputs[1]
        v = inputs[-1] if len(inputs) == 3 else k
        batch_size = q.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # note: attr split_size_or_sections indicates the size of the slice, not the num of the slice
        q = torch.cat(torch.split(q, split_size_or_sections=self.embed_size//self.multihead_num, dim=-1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.embed_size//self.multihead_num, dim=-1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.embed_size//self.multihead_num, dim=-1), dim=0)

        attention = torch.matmul(q, k.transpose(1, 2))/torch.sqrt(torch.tensor(self.embed_size
                                                                               //self.multihead_num).float())

        if mask is not None:
            assert mask.size().__len__() == 3

            mask = torch.tile(mask, dims=(self.multihead_num, 1, 1))
            attention -= 1e+9 * mask

        attention = torch.softmax(attention, dim=-1)

        feature = torch.matmul(attention, v)
        feature = torch.cat(torch.split(feature, split_size_or_sections=batch_size, dim=0), dim=-1)

        feature = self.linear(feature)
        feature = torch.dropout(feature, p=self.drop_rate, train=True)

        feature = torch.add(feature, inputs[0])

        feature = self.layer_norm(feature)

        return feature, attention

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name in ['layer_norm.weight', 'layer_norm.bias']:
                    continue
                elif name.split('.')[-1] == 'weight':
                    # glorot normal
                    stddev = math.sqrt(2/(sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)
                else:
                    torch.nn.init.zeros_(param)


class Affine(nn.Module):
    def __init__(self,
                 embed_size: int,
                 target_size: int):
        super(Affine, self).__init__()

        self.gamma_block = nn.Sequential(nn.Linear(in_features=embed_size,
                                                   out_features=target_size),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(in_features=target_size,
                                                   out_features=target_size))

        self.beta_block = nn.Sequential(nn.Linear(in_features=embed_size,
                                                  out_features=target_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=target_size,
                                                  out_features=target_size))

        self.init_params()

    def forward(self, x, y):

        weight = self.gamma_block(y)
        bias = self.beta_block(y)

        batch_size = x.size(0)
        weight = weight.view(batch_size, -1, 1, 1)
        bias = bias.view(batch_size, -1, 1, 1)

        feat = weight * x + bias

        return feat

    def init_params(self):

        for named_param in self.named_parameters():

            name, param = named_param
            """
            仿射层参数初始化有变动, 收藏项目并联系作者(m13541280433@163.com)获取
            """

class DF_Block(nn.Module):
    def __init__(self,
                 embed_size: int,
                 target_size: int):
        super(DF_Block, self).__init__()

        self.init_affine = Affine(embed_size=embed_size,
                                  target_size=target_size)
        self.final_affine = Affine(embed_size=embed_size,
                                   target_size=target_size)

    def forward(self, x, y):
        f = self.init_affine(x, y)
        f = F.leaky_relu(f, negative_slope=.2, inplace=True)

        f = self.final_affine(f, y)
        feat = F.leaky_relu(f, negative_slope=.2, inplace=True)

        return feat

class Spatial(nn.Module):
    def __init__(self,
                 features: int,
                 embed_size: int):
        super(Spatial, self).__init__()

        self.init_conv = nn.Conv2d(in_channels=features+embed_size,
                                   out_channels=features//4,
                                   kernel_size=(3, 3), padding=(1, 1),
                                   padding_mode='reflect')

        self.final_conv = nn.Conv2d(in_channels=features//4, out_channels=1,
                                    kernel_size=(4, 4))

        self.init_params()

    def forward(self, x, y):

        batch_size = y.size(0)
        height, width = x.size()[-2:]

        y = y.view(batch_size, -1, 1, 1)
        y = y.expand(-1, -1, height, width)

        f = torch.cat([x, y], dim=1)

        f = self.init_conv(f)
        f = F.leaky_relu(f, negative_slope=.2, inplace=True)

        feat = self.final_conv(f)

        return feat

    def init_params(self):

        for named_param in self.named_parameters():

            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    stddev = math.sqrt(2 / (sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)
                else:
                    torch.nn.init.zeros_(param)


class G_ResBlock(nn.Module):
    def __init__(self,
                 embed_size: int,
                 input_size: int,
                 target_size: int,
                 kernel_size: tuple,
                 padding: tuple,
                 padding_mode: str='reflect',
                 short_cut: bool=True,
                 upsample: bool=True,
                 upsample_mode: str='bicubic',
                 groups: int=1):
        super(G_ResBlock, self).__init__()
        assert not (input_size % groups or target_size % groups)
        assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular']
        assert upsample_mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
        self.input_size = input_size
        self.target_size = target_size
        self.upsample = upsample

        self.init_fuse = DF_Block(embed_size=embed_size, target_size=input_size)
        self.final_fuse = DF_Block(embed_size=embed_size, target_size=target_size//2 if short_cut else target_size)

        self.init_branch_conv = nn.Conv2d(in_channels=input_size,
                                          out_channels=target_size//2 if short_cut else target_size,
                                          kernel_size=kernel_size, padding=padding,
                                          padding_mode=padding_mode, groups=groups)

        self.final_branch_conv = nn.Conv2d(in_channels=target_size//2 if short_cut else target_size,
                                           out_channels=target_size,
                                           kernel_size=kernel_size, padding=padding,
                                           padding_mode=padding_mode, groups=groups)

        if input_size != target_size:
            self.main_conv = nn.Conv2d(in_channels=input_size, out_channels=target_size,
                                       kernel_size=(1, 1), groups=groups)

        self.init_batch_norm = nn.BatchNorm2d(num_features=target_size//2 if short_cut else target_size)
        self.final_batch_norm = nn.BatchNorm2d(num_features=target_size)

        if upsample:
            self.up_sample = nn.Upsample(scale_factor=(2, 2), mode=upsample_mode, align_corners=True)

        self.init_params()

    def forward(self, x, y):

        if self.upsample:
            x = self.up_sample(x)

        b = self.init_fuse(x, y)
        b = self.init_branch_conv(b)
        b = self.init_batch_norm(b)
        b = F.leaky_relu(b, negative_slope=.2, inplace=True)

        b = self.final_fuse(b, y)
        b = self.final_branch_conv(b)

        if self.input_size != self.target_size:
            x = self.main_conv(x)

        feat = x + b
        feat = self.final_batch_norm(feat)
        feat = F.leaky_relu(feat, negative_slope=.2, inplace=True)

        return feat

    def init_params(self):

        for named_param in self.named_parameters():

            name, param = named_param
            if param.requires_grad:
                if name.split('.')[0] not in ['init_fuse', 'final_fuse']:
                    if name.split('.')[-1] == 'weight':
                        stddev = math.sqrt(2 / (sum(param.size()[:2])))
                        torch.nn.init.normal_(param, mean=0, std=stddev)
                    else:
                        torch.nn.init.zeros_(param)


class D_ResBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 target_size: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,
                 padding_mode: str='reflect',
                 short_cut: bool=True,
                 downsample: bool=True,
                 groups: int=1):
        super(D_ResBlock, self).__init__()
        assert not (input_size % groups or target_size % groups)
        assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular']
        self.input_size = input_size
        self.target_size = target_size
        self.downsample = downsample

        self.init_branch_conv = nn.Conv2d(in_channels=input_size,
                                          out_channels=target_size//2 if short_cut else target_size,
                                          kernel_size=kernel_size, padding=padding,
                                          padding_mode=padding_mode, groups=groups,
                                          bias=False)

        self.final_branch_conv = nn.Conv2d(in_channels=target_size//2 if short_cut else target_size,
                                           out_channels=target_size,
                                           kernel_size=(3, 3), stride=stride,
                                           padding=padding, padding_mode=padding_mode,
                                           bias=False, groups=groups)

        self.gamma = nn.Parameter(torch.randn(size=()), requires_grad=True)

        if input_size != target_size:
            self.main_conv = nn.Conv2d(in_channels=input_size, out_channels=target_size,
                                       kernel_size=(1, 1), groups=groups)

        if downsample:
            self.down_sample = nn.AvgPool2d(kernel_size=(2, 2))

        self.init_params()

    def forward(self, x):

        b = self.init_branch_conv(x)
        b = F.leaky_relu(b, negative_slope=.2, inplace=True)

        b = self.final_branch_conv(b)

        if self.input_size != self.target_size:
            x = self.main_conv(x)

        if self.downsample:
            x = self.down_sample(x)

        feat = x + self.gamma * b
        feat = F.leaky_relu(feat, negative_slope=.2, inplace=True)

        return feat

    def init_params(self):

        for named_param in self.named_parameters():

            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    stddev = math.sqrt(2 / (sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)
                else:
                    torch.nn.init.zeros_(param)


if __name__ == '__main__':

    encoder = TextEncoder(
                 vocab_size=2000,
                 embed_size=128,
                 target_size=128,
                 multihead_num=4)

    # attn = MultiHeadAttention(
    #              source_size=64,
    #              embed_size=128,
    #              multihead_num=4,
    #              drop_rate=0.3)
    #
    # value = torch.randn(size=(4, 7, 64))
    #
    # feat, _ = attn([value,]*3)

    # value = torch.tensor(data=[[1, 2, 3, 4, 5, 0, 0, 0],
    #                            [1, 2, 3, 4, 5, 6, 0, 0],
    #                            [1, 2, 3, 4, 5, 6, 7, 0],
    #                            [1, 2, 3, 0, 0, 0, 0, 0]],
    #                      dtype=torch.int64)
    #
    # x = encoder(value)