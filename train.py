# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from DMFGan import DmfGan
from configs import config as cfg
from _utils.generate import DataGenerator

if __name__ == '__main__':

    ckpt_path = dict()
    if cfg.load_ckpt:
        ckpt_path.update({'gen_ckpt_path': cfg.ckpt_path['gen_ckpt_path'] + '\\模型文件',
                          'dis_ckpt_path': cfg.ckpt_path['dis_ckpt_path'] + '\\模型文件',
                          'txt_ckpt_path': cfg.ckpt_path['txt_ckpt_path'] + '\\模型文件'})

    dmfgan = DmfGan(block_num=cfg.block_num,
                   noise_size=cfg.noise_size,
                   embed_size=cfg.embed_size,
                   token_size=cfg.token_size,
                   target_size=cfg.target_size,
                   gen_in_channels=cfg.gen_in_channels,
                   gen_out_channels=cfg.gen_out_channels,
                   dis_in_channels=cfg.dis_in_channels,
                   dis_out_channels=cfg.dis_out_channels,
                   ckpt_path=ckpt_path,
                   load_ckpt=cfg.load_ckpt,
                   betas=cfg.betas,
                   weight_decay=cfg.weight_decay,
                   learning_rate=cfg.learning_rate)

    data_gen = DataGenerator(root_path=cfg.root_path,
                             token_size=cfg.token_size,
                             batch_size=cfg.batch_size,
                             train_ratio=cfg.train_ratio)

    train_gen = data_gen.data_generate(training=True)
    validate_gen = data_gen.data_generate(training=False)

    for epoch in range(cfg.Epoches):

        for i in range(data_gen.get_train_len()):
            text_srcs, former_srcs, latter_srcs = next(train_gen)
            noises = np.random.randn(text_srcs.shape[0], cfg.noise_size)
            dmfgan.train(noises, text_srcs, former_srcs, latter_srcs)
            if not (i + 1) % cfg.per_sample_interval:
                dmfgan.generate_sample(text_srcs, i + 1)

        torch.save({'gen_state_dict': dmfgan.generator.state_dict(),
                    'gf_loss': dmfgan.train_gf_loss / data_gen.get_train_len(),
                    'gl_loss': dmfgan.train_gl_loss / data_gen.get_train_len(),
                    'gf_acc': dmfgan.train_gf_acc / data_gen.get_train_len() * 100,
                    'gl_acc': dmfgan.train_gl_acc / data_gen.get_train_len() * 100},
                   cfg.ckpt_path['gen_ckpt_path'] + '\\Epoch{:0>3d}_gf_loss{:.2f}_gl_loss{:.2f}'
                                                    '_gf_acc{:.2f}_gl_acc{:.2f}.pth.tar'.format(
                       epoch + 1,
                       dmfgan.train_gf_loss / data_gen.get_train_len(),
                       dmfgan.train_gl_loss / data_gen.get_train_len(),
                       dmfgan.train_gf_acc / data_gen.get_train_len() * 100,
                       dmfgan.train_gl_acc / data_gen.get_train_len() * 100))

        torch.save({'dis_state_dict': dmfgan.discriminator.state_dict(),
                    'df_loss': dmfgan.train_df_loss / data_gen.get_train_len(),
                    'dl_loss': dmfgan.train_dl_loss / data_gen.get_train_len(),
                    'df_acc': dmfgan.train_df_acc / data_gen.get_train_len() * 100,
                    'dl_acc': dmfgan.train_dl_acc / data_gen.get_train_len() * 100},
                   cfg.ckpt_path['dis_ckpt_path'] + '\\Epoch{:0>3d}_df_loss{:.2f}_dl_loss{:.2f}'
                                                    '_df_acc{:.2f}_dl_acc{:.2f}.pth.tar'.format(
                       epoch + 1,
                       dmfgan.train_df_loss / data_gen.get_train_len(),
                       dmfgan.train_dl_loss / data_gen.get_train_len(),
                       dmfgan.train_df_acc / data_gen.get_train_len() * 100,
                       dmfgan.train_dl_acc / data_gen.get_train_len() * 100))

        torch.save({'txt_state_dict': dmfgan.text_encoder.state_dict()},
                   cfg.ckpt_path['txt_ckpt_path'] + '\\Epoch{:0>3d}_gf_loss{:.2f}_df_loss{:.2f}'
                                                    '_gl_loss{:.2f}_dl_loss{:.2f}.pth.tar'.format(
                       epoch + 1,
                       dmfgan.train_gf_loss / data_gen.get_train_len(),
                       dmfgan.train_df_loss / data_gen.get_train_len(),
                       dmfgan.train_gl_acc / data_gen.get_train_len() * 100,
                       dmfgan.train_dl_acc / data_gen.get_train_len() * 100))

        print(f'Epoch: {epoch + 1}\n'
              f'train_df_loss: {dmfgan.train_df_loss / data_gen.get_train_len()}\n'
              f'train_dl_loss: {dmfgan.train_dl_loss / data_gen.get_train_len()}\n'
              f'train_gf_loss: {dmfgan.train_gf_loss / data_gen.get_train_len()}\n'
              f'train_gl_loss: {dmfgan.train_gl_loss / data_gen.get_train_len()}\n'
              f'train_df_acc: {dmfgan.train_df_acc / data_gen.get_train_len() * 100}\n'
              f'train_dl_acc: {dmfgan.train_dl_acc / data_gen.get_train_len() * 100}\n'
              f'train_gf_acc: {dmfgan.train_gf_acc / data_gen.get_train_len() * 100}\n'
              f'train_gl_acc: {dmfgan.train_gl_acc / data_gen.get_train_len() * 100}\n')

        for i in range(data_gen.get_val_len()):

            text_srcs, former_srcs, latter_srcs = next(validate_gen)
            noises = np.random.randn(text_srcs.shape[0], cfg.noise_size)
            dmfgan.validate(noises, text_srcs, former_srcs, latter_srcs)

        print(f'val_df_loss: {dmfgan.val_df_loss / data_gen.get_val_len()}\n'
              f'val_dl_loss: {dmfgan.val_dl_loss / data_gen.get_val_len()}\n'
              f'val_gf_loss: {dmfgan.val_gf_loss / data_gen.get_val_len()}\n'
              f'val_gl_loss: {dmfgan.val_gl_loss / data_gen.get_val_len()}\n'
              f'val_df_acc: {dmfgan.val_df_acc / data_gen.get_val_len() * 100}\n'
              f'val_dl_acc: {dmfgan.val_dl_acc / data_gen.get_val_len() * 100}\n'
              f'val_gf_acc: {dmfgan.val_gf_acc / data_gen.get_val_len() * 100}\n'
              f'val_gl_acc: {dmfgan.val_gl_acc / data_gen.get_val_len() * 100}\n')

        # initialize
        dmfgan.train_gf_loss, dmfgan.val_gf_loss = 0, 0
        dmfgan.train_gl_loss, dmfgan.val_gl_loss = 0, 0
        dmfgan.train_df_loss, dmfgan.val_df_loss = 0, 0
        dmfgan.train_dl_loss, dmfgan.val_dl_loss = 0, 0
        dmfgan.train_gf_acc, dmfgan.val_gf_acc = 0, 0
        dmfgan.train_gl_acc, dmfgan.val_gl_acc = 0, 0
        dmfgan.train_df_acc, dmfgan.val_df_acc = 0, 0
        dmfgan.train_dl_acc, dmfgan.val_dl_acc = 0, 0
