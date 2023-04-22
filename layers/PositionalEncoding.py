# -*- coding: utf-8 -*-

# ***************************************************
# * File        : PositionalEncoding.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-22
# * Version     : 0.1.042223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch
from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PositionalEncoding(nn.Module):
    """
    Positional encoding.
    """
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype = torch.float32).reshape(-1, 1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype = torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
