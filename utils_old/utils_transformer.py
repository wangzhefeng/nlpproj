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

import torch
import torch.nn as nn

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

def clones(module, N):
    """
    Product N identical layers.
    """
    return nn.ModuleList([
        copy.deepcopy(module) 
        for _ in range(N)
    ])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape), 
        diagonal = 1
    ).type(torch.uint8)
    
    return subsequent_mask == 0




# 测试代码 main 函数
def main():
    import pandas as pd
    import altair as alt
    from utils.utils_transformer import show_example

    def example_mask():
        LS_data = pd.concat([
            pd.DataFrame({
                "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                "Window": y,
                "Masking": x,
            })
            for y in range(20)
            for x in range(20)
        ])

        return (
            alt.Chart(LS_data)
            .mark_rect()
            .properties(height = 250, width = 250)
            .encode(
                alt.X("Window:O"),
                alt.Y("Masking:O"),
                alt.Color("Subsequent Mask:Q", scale = alt.Scale(scheme = "viridis")),
            )
            .interactive()
        )

    show_example(example_mask)

if __name__ == "__main__":
    main()
