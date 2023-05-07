# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LSTM_tf.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
max_features = 20000
maxlen = 200
batch_size = 32
epochs = 2

# 数据加载
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(
    num_words = max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, 
    maxlen = maxlen
)
x_val = tf.keras.preprocessing.sequence.pad_sequences(
    x_val, 
    maxlen = maxlen
)


# 模型构建
def LSTM():
    inputs = tf.keras.Input(shape = (None,), dtype = tf.int32)
    x = keras.layers.Embedding(max_features, 128)(inputs)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences = True)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64)
    )(x)
    outputs = keras.layers.Dense(1, activation = "sigmoid")(x)
    return outputs



model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)
model.fit(
    x_train,
    y_train, 
    batch_size = batch_size, 
    epochs = epochs, 
    validation_data = (x_val, y_val)
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
