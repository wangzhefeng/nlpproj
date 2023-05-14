# -*- coding: utf-8 -*-

# ***************************************************
# * File        : loss.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-14
# * Version     : 0.1.051414
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

import altair as alt
import pandas as pd
import torch

from exp_transformer.LabelSmoothing import LabelSmoothing
from utils.utils_transformer import show_example

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([0, x / d, 1 / d, 1 / d, 1 / d])

    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )




# 测试代码 main 函数
def main():
    show_example(penalization_visualization)

if __name__ == "__main__":
    main()
