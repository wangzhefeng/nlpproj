# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MultiHeadedAttention.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-07
# * Version     : 0.1.050721
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_transformer import clones

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask = None, dropout = None):
        # score(dot product and scale)
        d_k = query.size(-1)
        #or scores = query @ key.transpose(-2, -1) / (d_k ** 0.5)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        p_attn = F.softmax(scores, dim = -1)
        # dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        #or return p_attn @ value, p_attn
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout = 0.1):
        """
        Take in model size and number of heads.

        W_{i}^{Q} \in R^{d_model \times d_{k}}
        W_{i}^{K} \in R^{d_model \times d_{k}}
        W_{i}^{V} \in R^{d_model \times d_{v}}
        W^{O} \in R^{h d_{v} \times d_model}
        which:
        h = 8
        dk=dv=d_model/h = 64
        
        Args:
            h (_type_): header nums [8]
            d_model (_type_): word embedding dim [512]
            dropout (float, optional): dropout. Defaults to 0.1.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # self.linears = nn.ModuleList([
        #     deepcopy(nn.Linear(d_model, d_model)) 
        #     for _ in range(4)
        # ])
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)
        self.attention = ScaledDotProductAttention()
        
    def forward(self, query, key, value, mask = None): 
        # batches
        num_batches = query.size(0)
        # mask
        if mask is not None:
            # same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [
            linear(x) \
                .view(num_batches, -1, self.h, self.d_k) \
                .transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self.attention(query, key, value, mask = mask, dropout = self.dropout)
        # 3) "Concat" using a view and apply a final linear 
        x = x \
            .transpose(1, 2) \
            .contiguous() \
            .view(num_batches, -1, self.h * self.d_k)
        # clear memory
        del query
        del key
        del value
        # Linear
        out = self.linears[-1](x)
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
