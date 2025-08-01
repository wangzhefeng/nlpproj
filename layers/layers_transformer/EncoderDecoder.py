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
LOGGING_LABEL = Path(__file__).name[:-3]


class EncoderDecoder(nn.Module):
    """
    编码器-解码器架构的基类
    """
    def __init__(self, encoder, decoder, src_embed, target_embed, generator) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator
    
    def forward(self, src, target, src_mask, target_mask):
        """
        take in and process masked src and target sequences.
        """
        enc_outputs = self.encode(src, src_mask)
        dec_outputs = self.decode(enc_outputs, src_mask, target, target_mask)
        return dec_outputs
    
    def encode(self, src, src_mask):
        return self.encoder(
            self.src_embed(src), 
            src_mask
        )
    
    def decode(self, memory, src_mask, target, target_mask):
        return self.decoder(
            self.target_embed(target), 
            memory, 
            src_mask, 
            target_mask
        )




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
