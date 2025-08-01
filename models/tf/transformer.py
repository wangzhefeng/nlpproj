# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_torch.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-07
# * Version     : 0.1.050718
# * Description : description
# * Link        : http://nlp.seas.harvard.edu/annotated-transformer/#part-3-a-real-world-example
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import copy

# import GPUtil
# import spacy
# import torch
import torch.nn as nn
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torchtext.datasets as datasets
# import torch.nn.functional as F
# from torch.nn.functional import log_softmax, pad
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torchtext.data.functional import to_map_style_dataset
# from torchtext.vocab import build_vocab_from_iterator
# Set to False to skip notebook execution (e.g. for debugging)
import warnings
warnings.filterwarnings("ignore")

from layers.transformer.MultiHeadAttention import MultiHeadAttention
from layers.transformer.PositionwiseFeedForward import PositionwiseFeedForward
from layers.transformer.PositionalEncoding import PositionalEncoding
from layers.transformer.EncoderDecoder import EncoderDecoder
from layers.transformer.Encoder import Encoder, EncoderLayer
from layers.transformer.Decoder import Decoder, DecoderLayer
from layers.transformer.Generator import Generator
from layers.transformer.Embeddings import Embeddings

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def transformer(src_vocab, target_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    """
    Helper: Construct a model from hyperparameters.

    Args:
        src_vocab (_type_): 输入序列长度
        target_vocab (_type_): 输出序列长度
        N (int, optional): Encoder 和 Decoder 的层数. Defaults to 6.
        d_model (int, optional): 输入序列 Embedding 后的长度. Defaults to 512.
        d_ff (int, optional): . Defaults to 2048.
        h (int, optional): Multi-head Self-Attention 的 head 数量. Defaults to 8.
        dropout (float, optional): dropout 的概率. Defaults to 0.1.
    """
    c = copy.deepcopy
    # multi-head attention
    attn = MultiHeadAttention(h = h, d_model = d_model)
    # position-wise Feed Forward Network
    ff = PositionwiseFeedForward(d_model = d_model, d_ff = d_ff, dropout = dropout)
    # position encoding
    position = PositionalEncoding(d_model = d_model, dropout = dropout)
    # encoder-decoder
    model = EncoderDecoder(
        Encoder(
            layer = EncoderLayer(
                size = d_model, 
                self_attn = c(attn), 
                feed_forward = c(ff), 
                dropout = dropout
            ), 
            N = N
        ),
        Decoder(
            layer = DecoderLayer(
                size = d_model, 
                self_attn = c(attn), 
                src_attn = c(attn), 
                feed_forward = c(ff), 
                dropout = dropout), 
            N = N
        ),
        nn.Sequential(Embeddings(d_model = d_model, vocab = src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model = d_model, vocab = target_vocab), c(position)),
        Generator(d_model = d_model, vocab = target_vocab),
    )
    # initialize parameters with Glorot / fan_avg.
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    return model



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
