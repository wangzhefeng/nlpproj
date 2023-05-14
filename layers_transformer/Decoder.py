# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DecoderLayer.py
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

import torch.nn as nn

from layers_transformer.LayerNorm import LayerNorm
from layers_transformer.SublayerConnection import SublayerConnection
from utils.utils_transformer import clones

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Decoder(nn.Module):
    """
    编码器-解码器架构的基本解码器接口
    N layer decoder with masking 
    """

    def __init__(self, layer, N) -> None:
        super(Decoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        # N stacked layers
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        # normalization
        out = self.norm(x)
        return out


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, target_mask):
        """
        decoder connections
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        out = self.sublayer[2](x, self.feed_forward)
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
