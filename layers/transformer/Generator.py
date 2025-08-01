# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Generator.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-07
# * Version     : 0.1.050718
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()

        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        x = self.proj(x)
        output = F.log_softmax(x, dim = -1)

        return output




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
