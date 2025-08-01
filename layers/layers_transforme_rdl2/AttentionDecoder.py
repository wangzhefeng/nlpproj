# -*- coding: utf-8 -*-

# ***************************************************
# * File        : AttentionDecoder.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-23
# * Version     : 0.1.042300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

from EncoderDecoder import Decoder


class AttentionDecoder(Decoder):
    """
    The base attention-based decoder interface.
    """
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
