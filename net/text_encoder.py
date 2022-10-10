import math
import torch
from torch import nn
from torch.nn.utils.rnn import (pad_packed_sequence,
                                pack_padded_sequence)
from layers.CustomLayers import MultiHeadAttention

class TextEncoder(nn.Module):
    def __init__(self,
                 token_size: int,
                 embed_size: int,
                 target_size: int,
                 multihead_num: int=4,
                 batch_first: bool=True,
                 bidirectional: bool=True,
                 drop_rate: float=.2):
        super(TextEncoder, self).__init__()
        assert not target_size % 2

        self.embedding = nn.Embedding(num_embeddings=token_size,
                                      embedding_dim=embed_size)

        self.attention = MultiHeadAttention(source_size=embed_size,
                                            embed_size=embed_size,
                                            multihead_num=multihead_num,
                                            drop_rate=drop_rate)

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=target_size//2 if bidirectional else target_size,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.init_params()

    def get_padding_mask(self, mask):

        pad_mask = torch.unsqueeze(torch.logical_not(mask).float(), dim=-1)
        pad_mask = torch.maximum(pad_mask, pad_mask.transpose(1, 2))

        return pad_mask

    def get_unique_mask(self, mask):
        """
        replaces pack_padded, pad_packed
        :param mask: bool array, sequence containing the padding elements
        :return: bool array, sequences of the last unpadded elements
        """
        batch_size = mask.size(0)

        masked_idx = mask.sum(dim=-1) - 1

        unique_mask = torch.zeros_like(mask)
        unique_mask[torch.arange(batch_size), masked_idx] = 1
        unique_mask = unique_mask.bool()

        return unique_mask

    def forward(self, sources):

        mask = sources.bool()
        padding_mask = self.get_padding_mask(mask)
        unique_mask = self.get_unique_mask(mask)

        x = self.embedding(sources)

        # multihead attention, feature enhancement
        feat, _ = self.attention([x, x, x], mask=padding_mask)

        feat, _ = self.lstm(feat)

        # this process can also be replaced by double indexing
        mask_out_feat = feat[unique_mask]

        return mask_out_feat

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[0] == 'attention':
                    continue
                elif name in ['embedding.weight', 'lstm.weight_ih_l0',
                              'lstm.weight_hh_l0', 'lstm.weight_ih_l0_reverse',
                              'lstm.weight_hh_l0_reverse']:
                    # glorot normal
                    stddev = math.sqrt(2/(sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)
                else:
                    torch.nn.init.zeros_(param)

    def get_weights(self):

        weights, bias = [], []
        for named_param in self.named_parameters():
            name, param = named_param
            if name in ['lstm.weight_ih_l0', 'lstm.weight_hh_l0',
                        'lstm.weight_ih_l0_reverse', 'lstm.weight_hh_l0_reverse']:
                weights.append(param)
            elif name in ['lstm.bias_ih_l0', 'lstm.bias_hh_l0',
                          'lstm.bias_ih_l0_reverse', 'lstm.bias_hh_l0_reverse']:
                bias.append(param)
            else:
                if name.split('.')[-1] == 'weight':
                    weights.append(param)
                else:
                    bias.append(param)

        return weights, bias
