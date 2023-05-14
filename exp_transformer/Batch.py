# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Batch.py
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

from utils.utils_transformer import subsequent_mask

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Batch:
    """
    Object for holding a batch of data with mask during training
    """
    
    def __init__(self, src, target = None, pad = 2) -> None:
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
     
    @staticmethod
    def make_std_mask(target, pad):
        """
        create a mask to hide padding and future words
        """
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(target_mask.data)
        return target_mask





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
