# -*- coding: utf-8 -*-

# ***************************************************
# * File        : inference_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-14
# * Version     : 0.1.051413
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

from transformer_torch import transformer
from utils.utils_transformer import (
    subsequent_mask, 
    show_example
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def inference():
    # model
    test_model = transformer(src_vocab = 11, target_vocab = 11, N = 2)
    test_model.eval()
    # input
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    # encoder
    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)
    # decoder
    for i in range(9):
        out = test_model.decode(
            memory, 
            src_mask, 
            ys, 
            subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim = 1)

    print(f"Example Untrained Model Prediction: {ys}")


def run_tests():
    for _ in range(10):
        inference()



# 测试代码 main 函数
def main():
    show_example(run_tests)

if __name__ == "__main__":
    main()
