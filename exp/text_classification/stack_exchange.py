# -*- coding: utf-8 -*-

# ***************************************************
# * File        : stack_exchange.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-01
# * Version     : 1.0.080116
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

import fasttext

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# data
data_dir = Path("./dataset/stack_exchange/")
train_data_dir = data_dir.joinpath("cooking.train")
valid_data_dir = data_dir.joinpath("cooking.valid")


# model
model_dir = Path("./saved_results/stack_exchange/model.bin")
model_dir.parent.mkdir(exist_ok=True, parents=True)
if not model_dir.exists():
    # model training
    model = fasttext.train_supervised(input=str(train_data_dir))
    # model save
    model.save_model(str(model_dir))
else:
    model = fasttext.load_model(str(model_dir))


# model test
test_res1 = model.predict("Which baking dish is best to bake a banana bread ?")
test_res2 = model.predict("Why not put knives in the dishwasher?")
logger.info(f"test_res1: {test_res1}")
logger.info(f"test_res2: {test_res2}")


# model evaluate
eval_res = model.test(str(valid_data_dir))
logger.info(f"eval_res: {eval_res}")



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
