# -*- coding:utf-8 -*-
"""
File Name: tmp
Version:
Description:
Author: liuxuewen
Date: 2017/10/20 14:32
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
# ——————————————————导入数据——————————————————————
# f=open('data_sin.csv')
from sklearn.preprocessing import MinMaxScaler

f = open('data/data_sinx.csv')
df = pd.read_csv(f)  # 从文件中读入数据
data = df.iloc[:, 2].values  # data=np.array(df['y'])   #获取y列的值

# normalize_data=(data-np.mean(data))/np.std(data)  #标准化
normalize_data = MinMaxScaler().fit_transform(data)  # .最小－最大规范化
# print(normalize_data.shape)
normalize_data = normalize_data.reshape([-1, 1])  # 维度转换
# print(normalize_data)

# 设置常量
time_step = 20  # 时间步
rnn_unit = 100  # hidden layer units
batch_size = 60  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率

# 训练集
train_x, train_y = [], []
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + time_step]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

# ——————————————————定义神经网络变量——————————————————
X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None, output_size])  # 每批次tensor对应的标签

# 输入层、输出层权重、偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, output_size]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_size, ]))
}


# ——————————————————定义模型——————————————————
def lstm(batch, output_keep_prob=0.8):  # 参数：输入网络批次数目
    # 输入维度由input的[-1,time_step,input_size]变为input_rnn的[-1,time_step,rnn_unit]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入

    # cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=output_keep_prob)#防止过拟合
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_2, output_keep_prob=output_keep_prob)#防止过拟合
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)  # 两层rnn

    init_state = cell.zero_state(batch, dtype=tf.float32)  # 初始化
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每次迭代（timestep）的结果，final_states是最后一次迭代的结果
    output = tf.reshape(output_rnn[:, -1], [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm():
    pred, _ = lstm(batch_size)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数初始化
        sess.run(tf.global_variables_initializer())
        # 参数恢复，从模型中读取
        # module_file = tf.train.latest_checkpoint('model_sin_2/')
        # saver.restore(sess, module_file)
        # 重复训练10000次
        step = 0
        while (True):
            step += 1
            start = random.randint(0, len(train_x) - batch_size)  # 从数据中随机选取起始点
            end = start + batch_size
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})

            # 每100步打印一次
            if step % 100 == 0:
                print(step, loss_)

            # 每2000步保存一次参数
            if step % 2000 == 0:
                print("保存模型：", saver.save(sess, 'model/sinx_model', global_step=step))

            # 当loss_小于某个值值，保存模型，退出
            if loss_ < 0.000001:
                print("over  保存模型：", saver.save(sess, 'model/sinx_model', global_step=step))
                break


# ————————————————预测模型————————————————————
def prediction():
    pred, _ = lstm(1, output_keep_prob=1)  # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model/')
        saver.restore(sess, module_file)

        # 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq = train_x[-1]
        predict = []
        # 得到之后200个预测结果
        for i in range(200):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            # print(next_seq,next_seq.shape)
            predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()


if __name__ == '__main__':
    train_lstm()
    # prediction()
