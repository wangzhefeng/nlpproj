# -*- coding: utf-8 -*-

# ***************************************************
# * File        : transformer_torch.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-07
# * Version     : 0.1.050718
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
import copy
import math
import time

import altair as alt
# import GPUtil
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchtext.datasets as datasets
import torch.nn.functional as F
# from torch.nn.functional import log_softmax, pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
# Set to False to skip notebook execution (e.g. for debugging)
import warnings
warnings.filterwarnings("ignore")

from utils_func import show_example

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def make_model(src_vocab, target_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    pass


def inference_test():
    pass



def run_tests():
    for _ in range(10):
        inference_test()




# 测试代码 main 函数
def main():
    show_example(run_tests)

if __name__ == "__main__":
    main()
