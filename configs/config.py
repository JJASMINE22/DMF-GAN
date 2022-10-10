# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch

# ===generator===
block_num=6
noise_size=100
embed_size=128
gen_in_channels=[1024, 512, 256, 256, 128, 128]
gen_out_channels=[512, 256, 256, 128, 128, 64]

# ===discriminator===
dis_in_channels=[64, 128, 128, 256, 256, 512]
dis_out_channels=[128, 128, 256, 256, 512, 1024]

# ===text encoder===
token_size=3000
target_size=128
multihead_num=4
batch_first=True
bidirectional=True
drop_rate=.3

# ===generate===
root_path = "C:\\DATASET\\birds"
token_path = "C:\\code\\DMF-GAN\\configs\\tokens.json"
batch_size = 4
train_ratio = 0.7
former_size = (64, 64)
latter_size = (256, 256)

flip_prob = 0.1
x_offset_range = (0, 20)
y_offset_range = (0, 20)
rotate_degrees = (-20, 20)

# ===train===
device = torch.device('cuda') if torch.cuda.is_available() else None
Epoches = 100
weight_decay = 5e-4
betas = {'beta1': 0.,
         'beta2': 0.9}
learning_rate = {'gen_lr': 1e-4,
                 'dis_lr': 1e-4,
                 'txt_lr': 1e-4}
ckpt_path = {'gen_ckpt_path': '绝对路径\\saved\\checkpoint\\generator',
             'dis_ckpt_path': '绝对路径\\saved\\checkpoint\\discriminator',
             'txt_ckpt_path': '绝对路径\\saved\\checkpoint\\text_encoder'}
per_sample_interval = 100
former_sample_path = '绝对路径\\samples\\former\\{}.jpg'
latter_sample_path = '绝对路径\\samples\\latter\\{}.jpg'
load_ckpt = True
