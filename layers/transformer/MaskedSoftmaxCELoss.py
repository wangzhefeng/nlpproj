# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MaskedSoftmaxCELoss.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-23
# * Version     : 0.1.042300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
from torch import nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def sequence_mask(X, valid_len, value=0):
    """
    Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`
    """
    maxlen = X.size(1)
    mask = torch.arange(
        (maxlen), 
        dtype = torch.float32,
        device = X.device
    )[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`
    """

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
