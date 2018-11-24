# -*- coding: utf-8 -*-
import numpy as np


def save(a, b):
    # a 是词向量 b是词典
    np.savetxt("词向量.csv", a, delimiter=',')
    with open("词.txt", 'w') as fp1:
        fp1.write('/'.join(b))
    return 1


def softmax(x):
    x = x.tolist()[0]
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0)) + 1e-30


def model(text):
    # text列表存放语料的分词结果
    # 返回词典和词向量
    window = 5  # 定义窗口大小，需要大于1
    alpha = 0.01
    iteration = 100  # 定义迭代轮数
    m = 50  # 定义词向量维度
    word = list(set(text))  # 词表
    print(len(word))
    wd = dict(zip(word, [i for i in range(len(word))]))  # 查找词位置的字典
    # 参数初始化
    vector = np.mat(np.random.random((len(word), m)) * 0.001)  # 词向量矩阵，维度为（词表长度，m）
    b = np.mat(np.random.random((1, m)))  # 隐藏层偏置
    r = np.mat(np.random.random((1, len(word))))  # 输出层偏置
    cweights = []  # 输入层到隐藏层的权重
    for i in range(window - 1):
        cweights.append(np.mat(np.random.random((m, m))))
    for item in range(iteration):
        start = 0
        while start + window < len(text):  # 遍历一遍语料
            context = text[start:start + window -1]  # 上文
            current = text[start + window - 1]  # 目标词
            p = [wd[item] for item in context]  # 上下文单词在词典中的位置
            pn = wd[current]  # 目标词在词典中的位置
            print('上文：', [word[pos] for pos in p], '预测目标词：', word[pn])
            # 前向传播
            s = np.dot(vector[p[0]], cweights[0].T)
            for i in range(1, len(p)):
                s += np.dot(vector[p[i]], cweights[i].T)
            h = s + b
            y = np.dot(h, vector.T) + r
            # 求解导数
            a = softmax(y)
            dy = np.mat([1 - a[i] if i == pn else -a[i] for i in range(len(a))])
            dh = np.dot(dy, vector)
            dcweights = []  # 输入层到隐藏层权重的导数
            for i in range(window - 1):
                dcweights.append(np.dot(dh.T, vector[p[i]]))
            # 对输入词向量求导，并更新
            dv = [] # 对输入上下文词向量的导数
            for i in range(window - 1):
                # i 代表上下文词语，p[i]代表这个词语在词典中的位置
                temp = np.dot(vector, cweights[i])
                # 在temp的第p[i]行，加上h
                temp[p[i]] += h
                vector[p[i]] += alpha * np.dot(dy, temp)
            # 更新网络其他参数
            b = b + alpha * dh
            r = r + alpha * dy
            for i in range(window - 1):
                cweights[i] = cweights[i] + alpha * dcweights[i]
            start += window
    save(vector, word)  # 保存词向量和词典
    return 1


with open('城市.txt', 'r', encoding='utf-8') as fp:
    test = '/'.join(fp.read().splitlines())
    test = test.split('/')
model(test)
