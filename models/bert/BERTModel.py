# -*- coding: utf-8 -*-

# ***************************************************
# * File        : BERTModel.py
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

from torch import nn

from layers.bert.BERTEncoder import BERTEncoder, MaskLM, NextSentencePred

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class BERTModel(nn.Module):
    """
    The BERT model.
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len = 1000, key_size = 768, query_size = 768, value_size = 768,
                 hid_in_features = 768, mlm_in_features = 768,
                 nsp_in_features = 768):
        super(BERTModel, self).__init__()

        self.encoder = BERTEncoder(
            vocab_size, num_hiddens, norm_shape,
            ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
            dropout, max_len = max_len, key_size = key_size,
            query_size=query_size, value_size=value_size
        )
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens), 
            nn.Tanh()
        )
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
