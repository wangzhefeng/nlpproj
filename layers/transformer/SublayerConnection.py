# -*- coding: utf-8 -*-

# ***************************************************
# * File        : SublayerConnection.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-07
# * Version     : 0.1.050719
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

from layers.transformer.LayerNorm import LayerNorm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class SublayerConnection(nn.Module):
    """
    A residual connection folled by a layer norm.
    """

    def __init__(self, size,  dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forwad(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
