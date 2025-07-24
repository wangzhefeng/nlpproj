# -*- coding: utf-8 -*-

# ***************************************************
# * File        : spacy_zh_model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-25
# * Version     : 1.0.072501
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

import spacy
import zh_core_web_sm, en_core_web_sm

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


nlp = spacy.load("zh_core_web_sm")
doc = nlp("这是一个用于示例的句子。")
for token in doc:
    print(f"token.text: {token.text}")
    print(f"token.pos_: {token.pos_}")
    print(f"token.dep_: {token.dep_}")
    print()


nlp = zh_core_web_sm.load()
doc = nlp("这是一个用于示例的句子。")
for token in doc:
    print(f"token.text: {token.text}")
    print(f"token.pos_: {token.pos_}")
    print(f"token.dep_: {token.dep_}")
    print()


nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sentence.")
for token in doc:
    print(f"token.text: {token.text}")
    print(f"token.pos_: {token.pos_}")
    print(f"token.dep_: {token.dep_}")
    print()


nlp = en_core_web_sm.load()
doc = nlp("This is a sentence.")
for token in doc:
    print(f"token.text: {token.text}")
    print(f"token.pos_: {token.pos_}")
    print(f"token.dep_: {token.dep_}")
    print()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
