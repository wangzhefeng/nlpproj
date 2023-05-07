# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-07
# * Version     : 0.1.050718
# * Description : Some convenience helper functions used throughout the notebook
# * Link        : http://nlp.seas.harvard.edu/annotated-transformer/#background
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
RUN_EXAMPLES = True


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):

    def __init__(self):
        self.param_groups = [{
            "lr": 0
        }]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none = False):
        None


class DummyScheduler:
    def step(self):
        None




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
