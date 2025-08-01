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
    """
    语料
    """
    
    def __iter__(self, text):
        # dictionary
        dictionary = corpora.Dictionary(text)
        # dictionary.save("/tmp/deerwester.dict")
        
        # corpus
        for line in open("https://radimrehurek.com/mycorpus.txt"):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())




# 测试代码 main 函数
def main():
    # TODO 构建预料库
    # corpus_memory_friendly = MyCorpus()
    # print(corpus_memory_friendly)
    # for vector in corpus_memory_friendly:
    #     print(vector)
    
    # 字典
    dictionary = corpora.Dictionary(
        line.lower().split() 
        for line in open("https://radimrehurek.com/mycorpus.txt")
    )
    print(dictionary)
    
    # 停止词
    stoplist = set("for a of the and to in".split(" "))
    stop_ids = [
        dictionary.token2id[stopword]
        for stopword in stoplist
        if stopword in dictionary.token2id
    ]
    print(stop_ids)

    # 频率统计
    once_ids = [
        token_id 
        for token_id, doc_freq in dictionary.dfs.items() 
        if doc_freq == 1
    ]
    print(once_ids)
    
    # 停止词和词频筛选
    dictionary.filter_tokens(stop_ids + once_ids)
    dictionary.compactify()
    print(dictionary)

if __name__ == "__main__":
    main()
