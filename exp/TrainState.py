# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TrainState.py
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

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TrainState:
    """
    Track number of steps, examples, and tokens processed
    """
    
    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total number of examples used
    tokens: int = 0  # total number of tokens processed




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
