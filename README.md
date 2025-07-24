# nlpproj

## Flow


## Task

* 不同数据语言(data language)
* 分词(Tokenization)
    - 将文本分割成单词、标点符号等
* 词性标注(Part-of-Speech-Tagging, POST)
    - 为词元分配词性，如动词或名词
* 依存句法分析(Dependency Parsing)
    - 为各个词元分配句法依存标签，描述它们之间的关系，如主语或宾语
* 词元还原(Lemmatization)
    - 为单词分配基本形式。例如，“was”的词元是“be”，“rats”的词元是“rat”
* 句子边界检测(Sentence Boundary Detection, SBD)
    - 识别并分割单个句子
* 句子分割(Sentence Segmentation)
* 命名实体识别(Named Entity Recognition, NER)
    - 标记现实世界中的命名对象，如人物、公司或地点
* 实体链接(Entity Linking, EL)
    - 将文本实体解析为知识库中的唯一标识符
* 文本分类(Text Classification)
    - 将类别或标签分配给整个文档或文档的一部分
* 相似度(Similarity)
    - 比较单词、文本片段和文档，以及它们彼此之间的相似程度
* 基于规则的匹配(Rule-based Matching)
    - 根据标记的文本和语言注释查找标记序列，类似于正则表达式
* 模型训练(Training)
    - 更新和改进统计模型的预测
* 序列化(Serialization)
    - 将对象保存到文件或字节字符串
* 形态分析(Morphological Analysis)
* ... 

## Tools

### spaCy

* [spaCy](https://github.com/explosion/spaCy)
* [spaCy models](https://github.com/explosion/spacy-models)
* [NeuralCoref](https://github.com/huggingface/neuralcoref)
* [textacy: NLP, before and after spaCy](https://textacy.readthedocs.io/en/stable/index.html)

1. 语言标注

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K startup for 1$ billion")
for token in doc:
    print(token.text, token.pos_, token.dep_)
```




### Gensim

