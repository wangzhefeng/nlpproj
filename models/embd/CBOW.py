# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CBOW.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-01
# * Version     : 1.0.080110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

from torch import nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


EMBED_DIMENSION = 512
EMBED_MAX_NORM = 1


class CBOW(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, voacab_size: int) -> None:
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings = voacab_size,
            embedding_dim = EMBED_DIMENSION,
            max_norm = EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features = EMBED_DIMENSION, 
            out_features = voacab_size
        )

    def forward(self, inputs_):
        x = self.embedding(inputs_)
        x = x.mean(axis = 1)
        x = self.linear(x)
        return x




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
