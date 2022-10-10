# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
from PIL import Image
from net.text_encoder import TextEncoder
from net.generator import Generator
from net.discriminator import Discriminator
from layers.CustomLosses import HingeLoss
from configs import config as cfg

class DmfGan:
    def __init__(self,
                 block_num: int,
                 noise_size: int,
                 embed_size: int,
                 token_size: int,
                 target_size: int,
                 gen_in_channels: list,
                 gen_out_channels: list,
                 dis_in_channels: list,
                 dis_out_channels: list,
                 ckpt_path: dict,
                 load_ckpt: bool,
                 betas: dict,
                 weight_decay: float,
                 learning_rate: dict):
        super(DmfGan, self).__init__()
        assert learning_rate.__len__() == 3
        assert all(iter(map(lambda key: isinstance(betas[key], float), betas)))
        assert all(iter(map(lambda key: isinstance(ckpt_path[key], str), ckpt_path)))
        assert all(iter(map(lambda key: isinstance(learning_rate[key], float), learning_rate)))

        self.generator = Generator(noise_size=noise_size,
                                   embed_size=embed_size,
                                   in_channels=gen_in_channels,
                                   out_channels=gen_out_channels,
                                   block_num=block_num)

        self.discriminator = Discriminator(combine_size=noise_size+embed_size,
                                           in_channels=dis_in_channels,
                                           out_channels=dis_out_channels,
                                           block_num=block_num)

        self.text_encoder = TextEncoder(token_size=token_size,
                                        embed_size=embed_size,
                                        target_size=target_size)

        if cfg.device:
            self.generator = self.generator.to(cfg.device)
            self.discriminator = self.discriminator.to(cfg.device)
            self.text_encoder = self.text_encoder.to(cfg.device)

        if load_ckpt:
            assert ckpt_path.__len__() == 3
            gen_ckpt_path, dis_ckpt_path, txt_ckpt_path = ckpt_path.values()
            gen_ckpt = torch.load(gen_ckpt_path)
            dis_ckpt = torch.load(dis_ckpt_path)
            txt_ckpt = torch.load(txt_ckpt_path)
            gen_ckpt_dict = gen_ckpt['gen_state_dict']
            dis_ckpt_dict = dis_ckpt['dis_state_dict']
            txt_ckpt_dict = txt_ckpt['txt_state_dict']

            self.generator.load_state_dict(gen_ckpt_dict)
            self.discriminator.load_state_dict(dis_ckpt_dict)
            self.text_encoder.load_state_dict(txt_ckpt_dict)

        self.loss_func = HingeLoss(reduction='mean')

        gen_lr, dis_lr, txt_lr = learning_rate.values()
        beta1, beta2 = betas.values()

        gen_former_weights, gen_former_bias, gen_latter_weights, gen_latter_bias = self.generator.get_weights()
        dis_former_weights, dis_former_bias, dis_latter_weights, dis_latter_bias = self.discriminator.get_weights()
        txt_weights, txt_bias = self.text_encoder.get_weights()
        # Generate, adversarial multiscale fusion, and text feature extraction, creating 5 optimizers
        self.GF_optimizer = torch.optim.Adam(params=[{'params': gen_former_weights, 'weight_decay': weight_decay},
                                                     {'params': gen_former_bias}], lr=gen_lr, betas=(beta1, beta2))
        self.GL_optimizer = torch.optim.Adam(params=[{'params': gen_latter_weights, 'weight_decay': weight_decay},
                                                     {'params': gen_latter_bias}], lr=gen_lr, betas=(beta1, beta2))
        self.DF_optimizer = torch.optim.Adam(params=[{'params': dis_former_weights, 'weight_decay': weight_decay},
                                                     {'params': dis_former_bias}], lr=dis_lr, betas=(beta1, beta2))
        self.DL_optimizer = torch.optim.Adam(params=[{'params': dis_latter_weights, 'weight_decay': weight_decay},
                                                     {'params': dis_latter_bias}], lr=dis_lr, betas=(beta1, beta2))
        self.E_optimizer = torch.optim.Adam(params=[{'params': txt_weights, 'weight_decay': weight_decay},
                                                    {'params': txt_bias}], lr=txt_lr)

        self.train_gf_loss, self.val_gf_loss = 0, 0
        self.train_gl_loss, self.val_gl_loss = 0, 0
        self.train_df_loss, self.val_df_loss = 0, 0
        self.train_dl_loss, self.val_dl_loss = 0, 0
        self.train_gf_acc, self.val_gf_acc = 0, 0
        self.train_gl_acc, self.val_gl_acc = 0, 0
        self.train_df_acc, self.val_df_acc = 0, 0
        self.train_dl_acc, self.val_dl_acc = 0, 0

    def train(self, noises, text_sources, former_sources, latter_sources):
        if cfg.device:
            noises = torch.tensor(noises, dtype=torch.float32).to(cfg.device)
            text_sources = torch.tensor(text_sources, dtype=torch.int64).to(cfg.device)
            former_sources = torch.tensor(former_sources, dtype=torch.float32).to(cfg.device)
            latter_sources = torch.tensor(latter_sources, dtype=torch.float32).to(cfg.device)
            fake_labels = torch.ones(size=(noises.size(0), 1)).to(cfg.device)
            total_labels = torch.cat([-torch.ones(size=(noises.size(0), 1)),
                                      torch.ones(size=(text_sources.size(0), 1))], dim=0).to(cfg.device)

        # ===Discriminator===
        """
        对抗器训练过程, 请通过收藏项目并联系作者(m13541280433@163.com)获取
        """

        dis_former_acc = torch.cat([torch.less_equal(fake_former_logits.squeeze(1), 0),
                                    torch.gt(real_former_logits.squeeze(1), 0)]).sum()/total_labels.size(0)
        dis_latter_acc = torch.cat([torch.less_equal(fake_latter_logits.squeeze(1), 0),
                                    torch.gt(real_latter_logits.squeeze(1), 0)]).sum()/total_labels.size(0)

        # ===Generator===
        """
        生成器训练过程, 请通过收藏项目并联系作者(m13541280433@163.com)获取
        """

        gen_former_acc = torch.gt(fake_former_logits.squeeze(1), 0).sum()/fake_labels.size(0)
        gen_latter_acc = torch.gt(fake_latter_logits.squeeze(1), 0).sum()/fake_labels.size(0)

        self.train_df_loss += D_latter_loss.data.item()
        self.train_dl_loss += D_former_loss.data.item()
        self.train_gf_loss += G_former_loss.data.item()
        self.train_gl_loss += G_latter_loss.data.item()

        self.train_df_acc += dis_former_acc.cpu().detach().numpy()
        self.train_dl_acc += dis_latter_acc.cpu().detach().numpy()
        self.train_gf_acc += gen_former_acc.cpu().detach().numpy()
        self.train_gl_acc += gen_latter_acc.cpu().detach().numpy()

    def validate(self, noises, text_sources, former_sources, latter_sources):
        if cfg.device:
            noises = torch.tensor(noises, dtype=torch.float32).to(cfg.device)
            text_sources = torch.tensor(text_sources, dtype=torch.int64).to(cfg.device)
            former_sources = torch.tensor(former_sources, dtype=torch.float32).to(cfg.device)
            latter_sources = torch.tensor(latter_sources, dtype=torch.float32).to(cfg.device)
            fake_labels = torch.ones(size=(noises.size(0), 1)).to(cfg.device)
            total_labels = torch.cat([-torch.ones(size=(noises.size(0), 1)),
                                      torch.ones(size=(text_sources.size(0), 1))], dim=0).to(cfg.device)

        # ===Discriminator===
        """
        对抗器验证过程, 请通过收藏项目并联系作者(m13541280433@163.com)获取
        """

        dis_former_acc = torch.cat([torch.less_equal(fake_former_logits.squeeze(1), 0),
                                    torch.gt(real_former_logits.squeeze(1), 0)]).sum() / total_labels.size(0)
        dis_latter_acc = torch.cat([torch.less_equal(fake_latter_logits.squeeze(1), 0),
                                    torch.gt(real_latter_logits.squeeze(1), 0)]).sum() / total_labels.size(0)

        # ===Generator===
        """
        生成器验证过程, 请通过收藏项目并联系作者(m13541280433@163.com)获取
        """

        gen_former_acc = torch.gt(fake_former_logits.squeeze(1), 0).sum() / fake_labels.size(0)
        gen_latter_acc = torch.gt(fake_latter_logits.squeeze(1), 0).sum() / fake_labels.size(0)

        self.val_df_loss += D_latter_loss.data.item()
        self.val_dl_loss += D_former_loss.data.item()
        self.val_gf_loss += G_former_loss.data.item()
        self.val_gl_loss += G_latter_loss.data.item()

        self.val_df_acc += dis_former_acc.cpu().detach().numpy()
        self.val_dl_acc += dis_latter_acc.cpu().detach().numpy()
        self.val_gf_acc += gen_former_acc.cpu().detach().numpy()
        self.val_gl_acc += gen_latter_acc.cpu().detach().numpy()

    def generate_sample(self, text_sources, batch):

        r, c = 1, 2
        try:
            random_index = np.random.choice(text_sources.shape[0], size=r*c, replace=False)
            text_sources = text_sources[random_index]
        except ValueError:
            pass
        else:
            if cfg.device:
                noises = torch.randn(size=(r * c, cfg.noise_size)).to(cfg.device)
                text_sources = torch.tensor(text_sources, dtype=torch.int64).to(cfg.device)

            text_features = self.text_encoder(text_sources)
            fake_former_sources, fake_latter_sources = self.generator(noises, text_features)
            fake_former_sources = np.clip(fake_former_sources.cpu().detach().numpy(), -1., 1.)
            fake_latter_sources = np.clip(fake_latter_sources.cpu().detach().numpy(), -1., 1.)
            fake_former_images = (fake_former_sources + 1) * 127.5
            fake_latter_images = (fake_latter_sources + 1) * 127.5
            fake_former_images = np.transpose(fake_former_images, [0, 2, 3, 1])
            fake_latter_images = np.transpose(fake_latter_images, [0, 2, 3, 1])

            former_samples = np.zeros(shape=(cfg.former_size[1], cfg.former_size[0]*2, 3))
            for i in range(r):
                for j in range(c):
                    former_samples[i * cfg.former_size[1]:(i + 1) * cfg.former_size[1],
                    j * cfg.former_size[0]:(j + 1) * cfg.former_size[0], :] = fake_former_images[i * c + j]

            former_samples = Image.fromarray(former_samples.astype('uint8'))
            former_samples.save(cfg.former_sample_path.format(batch), quality=95, subsampling=0)

            latter_samples = np.zeros(shape=(cfg.latter_size[1], cfg.latter_size[0]*2, 3))
            for i in range(r):
                for j in range(c):
                    latter_samples[i * cfg.latter_size[1]:(i + 1) * cfg.latter_size[1],
                    j * cfg.latter_size[0]:(j + 1) * cfg.latter_size[0], :] = fake_latter_images[i * c + j]

            latter_samples = Image.fromarray(latter_samples.astype('uint8'))
            latter_samples.save(cfg.latter_sample_path.format(batch), quality=95, subsampling=0)
