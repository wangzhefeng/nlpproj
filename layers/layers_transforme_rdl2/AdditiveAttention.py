# -*- coding: utf-8 -*-

# ***************************************************
# * File        : AdditiveAttention.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-23
# * Version     : 0.1.042300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch
from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def sequence_mask(X, valid_len, value = 0):
    """
    Mask irrelevant entries in sequences.
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype = torch.float32, device = X.device)[None, :] < valid_len[:, None]
    X[~mask] = value

    return X


def masked_softmax(X, valid_lens):
    """
    Perform softmax operation by masking elements on the last axis.
    
    Args:
        X (_type_): 3D tensor
        valid_lens (_type_): 1D or 2D tensor

    Returns:
        _type_: _description_
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim = -1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements 
        # with a very large negative value, 
        # whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value = -1e6)

        return nn.functional.softmax(X.reshape(shape), dim = -1)


class AdditiveAttention(nn.Module):
    """
    Additive attention.
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias = False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias = False)
        self.w_v = nn.Linear(num_hiddens, 1, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
