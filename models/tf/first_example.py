# -*- coding: utf-8 -*-

# ***************************************************
# * File        : first_example.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-05-08
# * Version     : 0.1.050822
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

from exp.utils_transformer import subsequent_mask
from layers.transformer.LabelSmoothing import LabelSmoothing

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Batch:
    """
    Object for holding a batch of data with mask during training
    """
    
    def __init__(self, src, target = None, pad = 2) -> None:
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
     
    @staticmethod
    def make_std_mask(target, pad):
        """
        create a mask to hide padding and future words
        """
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(target_mask.data)
        return target_mask


def data_gen(V, batch_size, num_batches):
    """
    Generate random data for a src-target copy task
    """
    for i in range(num_batches):
        data = torch.randint(1, V, size = (batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        target = data.requires_grad_(False).clone().detach()
        yield Batch(src, target, 0)


class SimpleLossCompute:
    """
    A simple loss compute and train function
    """

    def __init__(self, generator, criterion) -> None:
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# Train the simple copy task.
def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size = V, padding_idx = 0, smoothing = 0.0)
    model = make_model(V, V, N = 2)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.5, betas = (0.9, 0.98), eps = 1e-9)
    lr_scheduler = LambdaLR(
        optimizer = optimizer,
        lr_lambda = lambda step: rate(step, model_size = model.src_embed[0].d_model, factor = 1.0, warmup = 400),
    )
    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

# execute_example(example_simple_model)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
