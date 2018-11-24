import numpy as np

vec = np.loadtxt('词向量.csv',dtype=float, delimiter=',')
with open('词表.txt', 'r', encoding = 'gbk') as fp1:
    word = fp1.read().split('/')
    print(word)
pos = [i for i in range(len(word))]
wd = dict(zip(word, pos))


def distance(a, b):
    # 返回2个单词的欧式距离
    vec1 = vec[wd[a]]
    vec2 = vec[wd[b]]
    return np.linalg.norm(vec1 - vec2)

def cov(a, b):
    # 返回2个单词的余弦夹角。
    a = vec[wd[a]]
    b = vec[wd[b]]
    r = np.dot(a, b)
    sa = pow(np.sum(np.dot(a, a)), 0.5)
    sb = pow(np.sum(np.dot(b, b)), 0.5)
    return r/(sa*sb)


def find(t):
    c = []
    for w in word:  # 遍历单词列表，找到与目标词最接近的10个词
        c.append(cov(t, w))
    d = dict(zip(word, c))
    n = sorted(d.items(), key = lambda item:item[1], reverse= True)
    for i in range(10):
        print(n[i])
    return 1


t = '南京'
print('与 "', t, '" 最相关的词：')
find(t)
