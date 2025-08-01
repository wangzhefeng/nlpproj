# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-14
# * Version     : 0.1.051414
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
import time

from exp_transformer.TrainState import TrainState

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def run_epoch(data_iter,
              model,
              loss_compute,
              optimizer,
              scheduler,
              mode = "train",
              accum_iter = 1,
              train_state = TrainState()):
    """
    Train a single epoch
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.target, batch.src_mask, batch.target_mask)
        loss, loss_node = loss_compute(out, batch.target_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print((
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    
    return total_loss / total_tokens, train_state




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
