import numpy as np
from collections import Counter
import random
import datetime


def wordcounter(text):
    # 返回词典、词典位置、词频。
    with open(text, 'r', encoding="utf-8") as fp:
        res = fp.read().splitlines()
    res = '/'.join(res)
    doc = res.split('/')  # doc 是由词组成的语料，类型是列表
    voc = list(set(doc))  # voc 是词典。
    freq = Counter(doc)  # freq是个字典，保存词频
    freq_dict = dict(zip(voc, [pow(freq[item], 0.75) for item in voc]))
    freq_length = sum(freq_dict.values())
    return doc, voc, freq, freq_length, freq_dict


def negtive_sampling(word, freq, v, length, fd):
    # word 是窗口内容，freq词频，v词表
    # 返回一个列表，包含window个负采样到的词
    neg = []
    m = pow(10, 8)
    for i in range(len(word) - 1):
        sum_len = 0
        r = random.randint(1, m) * length / m
        for item in v:
            sum_len += fd[item]
            if r < sum_len:
                if item not in word:
                    neg.append(item)
                    break
    return neg


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def skip_gram(text):  # 基于负采样算法的实现
    # 输入text是分好词的列表。词之间用'/'隔开，可以有换行。
    # 例如 text=
    # '我/以前/在/东京/打/篮球/但/我的/技术/没有/很好
    # 中国/的/上海/美丽/城市/我/喜欢/去/上海/旅行/我/昨天/坐/火车/到达/上海/火车站
    # 返回值：vector是词向量矩阵，vocabulary是词汇列表，用'/'分割。
    corpus, vocabulary, freq, freq_length, freq_dict = wordcounter(text)
    pos = [i for i in range(len(vocabulary))]
    v_dict = dict(zip(vocabulary, pos))  # v_dict是个字典，给每个词编一个独立的号。
    index = 50  # 词向量的维度
    w = np.random.random((len(vocabulary), index))  # w的每一行是一个参数向量
    vector = np.random.random((len(vocabulary), index))
    alpha = 0.01
    window = 3  # 窗口大小，最小为2。
    print('语料长度：', len(corpus), '词典大小：', len(vocabulary))
    for iters in range(100):
        start = 0
        while start+window <= len(corpus):
            current = corpus[start + int(window/2)]
            context = [corpus[temp] for temp in range(start, start+window)]
            neg_s = negtive_sampling(context, vocabulary, freq, freq_length, freq_dict)
            context.pop(int(window/2))
            u = [v_dict[item] for item in neg_s+context]
            L = lambda x: x >= window - 1  # 前几个是负采样的单词 定义lambda函数判断样本正负
            e = 0
            for i in range(len(u)):
                q = sigmoid(np.dot(vector[v_dict[current]], w[u[i]].T))
                g = alpha * (L(i) - q)
                e = e + g * w[u[i]]
                w[u[i]] += g * vector[v_dict[current]]
                vector[v_dict[current]] += e
            start += window
    return vector, vocabulary  # word 是词向量矩阵，v是词表


print('修改时间：', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
word, v = skip_gram('城市.txt')
np.savetxt('词向量.csv', word, delimiter=',')
with open('词表.txt', 'w') as fp:
    fp.write('/'.join(v))
print('保存参数！')
print('结束时间：', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


