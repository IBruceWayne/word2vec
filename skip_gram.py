# -*- coding: utf-8 -*-
# @author: gcg
# 使用tensorflow实现skip-gram模型，损失函数使用NCE

import tensorflow as tf
from collections import Counter
import numpy as np
import time
# parameters
VOCAB_SIZE = 800
EMBED_SIZE = 300
BATCH_SIZE = 100
SKIP_WINDOW = 2  # the context window
NUM_SAMPLED = 10  # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 500


def read_data(file_path):
    # 读取数据返回语料内容
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().split()
    return words


def build_vocab(words, vocab_size):
    # 创建单词和下标的映射
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = index
        index += 1
    return dictionary


def convert_words_to_index(words, dictionary):
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = np.random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target


def gen():
    # 返回中心词cent_word 上下文target_word
    words = read_data('语料.txt')
    dictionary = build_vocab(words, VOCAB_SIZE)
    with open('dict.txt', 'w', encoding='utf-8') as fp:
        for a, b in dictionary.items():
            fp.write(a + ' ' + str(b) + '\n')
    index_words = convert_words_to_index(words, dictionary)
    del words  # 节省内存
    # 在index_words上滑动窗口取词
    single_gen = generate_sample(index_words, SKIP_WINDOW)
    while True:
        center_batch = np.zeros(BATCH_SIZE, dtype=np.int32)
        target_batch = np.zeros([BATCH_SIZE, 1])
        for index in range(BATCH_SIZE):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch


def distance(a, b):
    # 返回2个单词的欧式距离
    return np.linalg.norm(a - b)

def cov(a, b):
    # 返回2个单词的余弦夹角。
    r = np.dot(a, b)
    sa = pow(np.sum(np.dot(a, a)), 0.5)
    sb = pow(np.sum(np.dot(b, b)), 0.5)
    return r/(sa*sb)

def test(m):
    word = []
    index = []
    with open('dict.txt', 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            a, b = line.strip('\n').split()
            word.append(a)
            index.append(int(b))
    dictionary = dict(zip(word, index))
    test_word = input('输入测试单词：')
    count = []
    for item in word:
        count.append(cov(m[dictionary[item]], m[dictionary[test_word]]))

    d = dict(zip(word, count))
    n = sorted(d.items(), key=lambda item: item[1], reverse=True)
    for i in range(10):
        print(n[i])

with tf.name_scope('data'):
    dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    iterator = dataset.make_initializable_iterator()
    center_words, target_words = iterator.get_next()

with tf.name_scope("embedding"):
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE]))
    embed = tf.nn.embedding_lookup(embed_matrix, center_words)

with tf.name_scope("loss"):
    nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE],
                                 initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
    nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                         biases=nce_bias,
                                         labels=target_words,
                                         inputs=embed,
                                         num_sampled=NUM_SAMPLED,
                                         num_classes=VOCAB_SIZE), name='loss')
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    total_loss = 0
    for index in range(NUM_TRAIN_STEPS):
        try:
            batch_loss, _, embedding= sess.run([loss, optimizer, embed_matrix])
            total_loss += batch_loss
            if (index + 1) % SKIP_STEP == 0:
                test(embedding)
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)







