# -*- coding: utf-8 -*-

# ***************************************************
# * File        : example.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-14
# * Version     : 0.1.051421
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

import torch

from exp_transformer.Batch import Batch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_gen(V, batch_size, num_batches):
    """
    Generate random data for a src-target copy task
    """
    for i in range(num_batches):
        data = torch.randint(1, V, size = (batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        target = data.requires_grad_(False).clone().detach()
        yield Batch(src, target, 0)


class SimpleLossCompute:
    """
    A simple loss compute and train function
    """

    def __init__(self, generator, criterion) -> None:
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), 
                y.contiguous().view(-1)
            ) / norm
        )
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
