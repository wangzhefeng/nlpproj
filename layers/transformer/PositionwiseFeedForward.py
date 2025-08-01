# -*- coding: utf-8 -*-

# ***************************************************
# * File        : PositionwiseFeedForward.py
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
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class PositionwiseFeedForward(nn.Module):
    """
    Implements Feed-Forward Networks
    FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}
    """

    def __init__(self, d_model, d_ff, dropout = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
