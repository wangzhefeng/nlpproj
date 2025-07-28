# -*- coding: utf-8 -*-

# ***************************************************
# * File        : my_corpus.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072815
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

from gensim import corpora
from smart_open import open

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class MyCorpus:
    
    def __iter__(self, text):
        # dictionary
        dictionary = corpora.Dictionary(text)
        dictionary.save("/tmp/deerwester.dict")
        # corpus
        for line in open("https://radimrehurek.com/mycorpus.txt"):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
