# -*- coding: utf-8 -*-

# ***************************************************
# * File        : layer_clones.py
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

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import copy

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def clones(module, N):
    """
    Product N identical layers.
    """
    return nn.ModuleList([
        copy.deepcopy(module) 
        for _ in range(N)
    ])




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
