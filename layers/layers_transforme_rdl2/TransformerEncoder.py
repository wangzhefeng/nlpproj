# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TransformerEncoder.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-23
# * Version     : 0.1.042300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import math
from torch import nn

from PositionalEncoding import PositionalEncoding
from EncoderBlock import EncoderBlock
from EncoderDecoder import Encoder

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class TransformerEncoder(Encoder):
    """
    Transformer encoder.
    """
    def __init__(self, 
                 vocab_size, 
                 key_size, 
                 query_size, 
                 value_size,
                 num_hiddens, 
                 norm_shape, 
                 ffn_num_input, 
                 ffn_num_hiddens,
                 num_heads, 
                 num_layers, 
                 dropout, 
                 use_bias = False, 
                 **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                EncoderBlock(
                    key_size, 
                    query_size, 
                    value_size, 
                    num_hiddens,
                    norm_shape, 
                    ffn_num_input, 
                    ffn_num_hiddens,
                    num_heads, 
                    dropout, 
                    use_bias,
                )
            )

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
