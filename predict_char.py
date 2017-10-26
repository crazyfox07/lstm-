# -*- coding:utf-8 -*-
"""
File Name: predict_char
Version:
Description:
Author: liuxuewen
Date: 2017/10/25 16:10
"""
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
from tensorflow.contrib import legacy_seq2seq as seq2seq


# 使用方法，训练命令：python predict_char 0,测试命令：python predict_char 1
class HParam():
    batch_size = 64
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    state_size = 256  # 节点数
    num_layers = 2
    seq_length = 10
    metadata = 'data/poets/poet.csv'
    gen_num = 500  # how many chars to generate


class DataGenerator():
    def __init__(self, datafiles, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        with open(datafiles, encoding='utf-8') as f:
            self.data = f.read()

        self.total_len = len(self.data)  # total data length
        self.words = list(set(self.data))
        self.words.sort()
        # vocabulary
        self.vocab_size = len(self.words)  # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}

        # pointer position to generate current batch
        self._pointer = 0

        # save metadata file
        self.save_metadata(args.metadata)

    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, id):
        return self.id2char_dict[id]

    def save_metadata(self, file):
        with open(file, 'w') as f:
            f.write('id\tchar\n')
            for i in range(self.vocab_size):
                c = self.id2char(i)
                # print(i,c,type(c))
                try:
                    f.write('{}\t{}\n'.format(i, c))
                except:
                    print(c, repr(c), '1111111111111')
                    pass

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len:
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer +
                           1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length  # update pointer position

            # convert to ids
            bx = [self.char2id(c) for c in bx]
            by = [self.char2id(c) for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches


class Model():
    """
    The core recurrent neural network model.
    """

    def __init__(self, args, data, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(
                tf.int32, [args.batch_size, args.seq_length])
            self.target_data = tf.placeholder(
                tf.int32, [args.batch_size, args.seq_length])

        with tf.name_scope('model'):
            self.cell = MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(
                BasicLSTMCell(args.state_size), output_keep_prob=0.7)] * args.num_layers)
            self.initial_state = self.cell.zero_state(
                args.batch_size, tf.float32)
            with tf.variable_scope('rnnlm'):
                w = tf.get_variable(
                    'softmax_w', [args.state_size, data.vocab_size])
                b = tf.get_variable('softmax_b', [data.vocab_size])
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [data.vocab_size, args.state_size])
                    self.embedding = embedding
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
                    self.in_data = self.input_data
                    self.inputs = inputs
            outputs, last_state = tf.nn.dynamic_rnn(
                self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(outputs, [-1, args.state_size])

            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)])
            self.cost = tf.reduce_sum(loss) / args.batch_size
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.summary.scalar('learning_rate', self.lr)

            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.summary.histogram(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()


def train(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化参数
        sess.run(tf.global_variables_initializer())

        # 从模型中读取参数
        # ckpt = tf.train.latest_checkpoint(args.log_dir)
        # saver.restore(sess, ckpt)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'rnnlm/embedding:0'
        embed.metadata_path = args.metadata

        max_iter = args.n_epoch * \
                   (data.total_len // args.seq_length) // args.batch_size

        i = 0
        while True:
            i = i + 1
            learning_rate = args.learning_rate * \
                            (args.decay_rate ** (i // args.decay_steps))
            x_batch, y_batch = data.next_batch()
            feed_dict = {model.input_data: x_batch,
                         model.target_data: y_batch, model.lr: learning_rate}
            train_loss, summary, _, _ = sess.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                 feed_dict)

            if i % 100 == 0:
                # writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i,
                                                               max_iter, train_loss))
            if i % 5000 == 0:
                saver.save(sess, 'model_poet/model_poet', global_step=i)
            if train_loss < 1:
                saver.save(sess, 'model_poet/model_poet', global_step=i)
                break


def sample(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint('model_poet')
        saver.restore(sess, ckpt)
        prime = '我们'
        state = sess.run(model.cell.zero_state(1, tf.float32))

        for word in prime:
            x = np.zeros((1, 1))
            x[0, 0] = data.char2id(word)

            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)


        word = prime[-1]
        lyrics = prime
        char_num = random.randint(100, 300)
        for i in range(200):
            x = np.zeros([1, 1])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(np.argmax(p))
            print(word, end='')
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += word
        with open('t2.txt', 'w',encoding='utf-8') as f:
            f.write(lyrics)
        return lyrics


def main(infer):
    args = HParam()
    # data = DataGenerator('JayLyrics.txt', args)
    data = DataGenerator('data/data_char.txt', args)
    model = Model(args, data, infer=infer)

    run_fn = sample if infer else train

    run_fn(data, model, args)



if __name__ == '__main__':

    if len(sys.argv) == 2:
        infer = int(sys.argv[-1])
        print('--Sampling--' if infer else '--Training--')
        main(infer)
    else:
        sys.exit(1)
