# -*- coding: utf-8 -*-

# ***************************************************
# * File        : EncoderLayer.py
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


class Encoder(nn.Module):
    """
    编码器-解码器架构的基本编码器接口
    a stack of N layers
    """
    def __init__(self, layer, N) -> None:
        super(Encoder, self).__init__()
        # N stacked layers
        self.layers = clones(layer, N)
        # normalization layer
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn 
        """
        # N stacked layers
        for layer in self.layers:
            x = layer(x, mask)
        # normalization
        out = self.norm(x)
        return out


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        Encoder connections
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        out = self.sublayer[1](x, self.feed_forward)
        return out




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
