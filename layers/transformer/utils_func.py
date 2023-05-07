# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils_layer.py
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

import torch
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


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape), 
        diagonal = 1
    ).type(torch.uint8)




# 测试代码 main 函数
def main():
    import pandas as pd
    import altair as alt
    from models.transformer.utils_func import show_example

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
