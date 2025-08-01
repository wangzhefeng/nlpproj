# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Embeddings.py
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

import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Embeddings(nn.Module):
    
    def __init__(self, d_model, vocab) -> None:
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        out = self.lut(x) * math.sqrt(self.d_model)
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
