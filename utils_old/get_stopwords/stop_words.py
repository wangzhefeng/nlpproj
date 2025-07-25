# -*- coding: utf-8 -*-

# ***************************************************
# * File        : stop_words.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-25
# * Version     : 1.0.072511
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

import nltk

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def download_stopwords(path: str=None):
    """
    停止词下载
    """
    # TODO
    # data_path = Path("./dataset") if path is None else Path(path)
    # nltk.download("stopwords", download_dir=data_path)
    nltk.download("stopwords")


def para_fraction(text, lang: str="chinese"):
    """
    TODO

    Args:
        text (_type_): _description_
        lang (str, optional): _description_. Defaults to "en".

    Returns:
        _type_: _description_
    """
    stopwords = nltk.corpus.stopwords.words(lang)
    para = [w for w in text if w.lower() not in stopwords]

    return len(para) / len(text)




# 测试代码 main 函数
def main():
    # download
    download_stopwords()

    # stop words
    stopwords_en = nltk.corpus.stopwords.words("english")
    logger.info(f"english stop word:\n{stopwords_en}")
 
    stopwords_zh = nltk.corpus.stopwords.words("chinese")
    logger.info(f"chinese stop word:\n{stopwords_zh}")

if __name__ == "__main__":
    main()
