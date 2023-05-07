# -*- coding: utf-8 -*-

# ***************************************************
# * File        : EncoderDecoder.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032717
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

from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class EncoderDecoder(nn.Module):
    """
    编码器-解码器架构的基类
    """
    def __init__(self, encoder, decoder, src_embed, target_embed, generator, **kwargs) -> None:
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator
    
    def forward(self, src, target, src_mask, target_mask, *args):
        """
        take in and process masked src and target sequences.

        Args:
            src (_type_): 输入源序列
            target (_type_): 输出目标序列
            src_mask (_type_): _description_
            target_mask (_type_): _description_

        Returns:
            _type_: # ???
        """
        enc_outputs = self.encode(src, src_mask, *args)
        dec_outputs = self.decode(enc_outputs, src_mask, target, target_mask, *args)
        return dec_outputs
    
    def encode(self, src, src_mask):
        """
        # ???

        Args:
            src (_type_): _description_
            src_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, target, target_mask):
        """
        # ???

        Args:
            memory (_type_): _description_
            src_mask (_type_): _description_
            target (_type_): _description_
            target_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.deocder(self.target_embed(target), memory, src_mask, target_mask)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
