# -*- encoding:utf8 -*-
import torch
import torch.autograd as autograd       # torch中自动计算梯度模块
import torch.nn as nn                   # 神经网络模块
import torch.optim as optim             # 模型优化器模块
from lstmModel import LSTMClassifier
import Utils as u
import numpy as np

torch.manual_seed(1)

# ****************************参数设置**************************** #
EMBEDDING_DIM = 100                         # 词向量维度
HIDDEN_DIM = 100                            # LSTM隐藏层维度
EPOCH = 1000                                # 训练次数
EARLY_STOP = True                           # 是否启用early stop
EARLY_STOP_THRESHOLD = 4                    # early stop的阈值
LEARNING_RATE = 0.001                       # 学习率
VALID_RATE = 0.4                            # 验证集占比
TEST_RATE = 0.2                             # 测试集占比
TRAIN_TIMES = 10                            # 需要的模型总数
LOG_DIR = '../logs/tag1-classifier.txt'     # 日志目录
DATA_DIR = '../data/alldata(fixed1).pkl'    # 数据目录
TAG_DIR = '../data/tag12.pkl'               # 分类标签来源
# *************************************************************** #


data = u.loadPickle(DATA_DIR)  # 载入数据集
tag12 = u.loadPickle(TAG_DIR)  # 载入分类信息


# 按照测试集/数据集=rate的比例打乱数据集
def shuffleData(data, rate):
    nsamples = len(data)

    sidx = np.random.permutation(nsamples)

    ntrain = int(np.round(nsamples * (1.0 - rate)))

    train_data = [data[s] for s in sidx[:ntrain]]
    test_data = [data[s] for s in sidx[ntrain:]]

    return train_data, test_data


# 将原始输入处理成torch接受的格式
def preparexy(seq, word2ix, tag2ix):
    idxs = [word2ix[w] for w in seq[0].split()]
    x = idxs
    y = tag2ix[seq[1]]
    return x, y


def getWord2Ix(data):
    word2ix = {}  # 单词的索引字典
    for sent, tag1, _, _ in data:
        for word in sent.split():
            if word not in word2ix:
                word2ix[word] = len(word2ix)
    return word2ix


# 该函数只能抽取一级分类
def getTag2Index(tags):
    tag2ix = {}  # 类别的索引字典
    for key in tags:
        if key not in tag2ix:
            tag2ix[key] = len(tag2ix)
    return tag2ix


# 计算模型准确率
def evaluate(data, word2ix, tag2ix, model):
    count = .0  # 统计正确分类的样本数
    total = .0  # 统计样本总数
    for i in range(len(data)):
        if data[i][1].strip() == '':
            continue
        testx, testy = preparexy(data[i], word2ix, tag2ix)
        testx = autograd.Variable(torch.LongTensor(testx))
        testout = model(testx)

        predy = torch.max(testout, 1)[1].data.numpy().squeeze()
        if predy == testy:
            count += 1.0
        total += 1.0
    return np.round(count / total, 4)


# 训练
def train_step(data, word2ix, tag2ix, model, epoch):
    for i in range(len(data)):
        # 如果没有标签，就直接跳过
        if data[i][1].strip() == '':
            continue
        if i % 500 == 0:
            print '第%d轮, 第%d个样本' % (epoch + 1, i + 1)

        # 得到输入和标签
        x, y = preparexy(data[i], word2ix, tag2ix)

        x = autograd.Variable(torch.LongTensor(x))
        y = autograd.Variable(torch.LongTensor([y]))

        out = model(x)  # (1 × 类别数目)

        loss = loss_function(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


word2ix = getWord2Ix(data)  # 单词索引字典
tag2ix = getTag2Index(tag12)  # 类别索引字典

vocab_size = len(word2ix)  # 27003
tags_size = len(tag2ix)  # 5

for time in range(TRAIN_TIMES):
    # 定义模型
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, tags_size)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 定义参数优化方法
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 打散数据集合，其中20%用于测试，训练中不使用

    training_data, test_data = shuffleData(data, TEST_RATE)  # 5600 1400 其中没有标签的有320个

    early_stop_count = 0    # 统计验证集上准确率连续没有提高的次数
    pre_accurary = 0.0      # 记录该次训练之前模型最好的准确率

    flag = 'normal'         # 是否正常完成训练

    for epoch in range(EPOCH):

        # 每轮训练都打乱数据集，分为训练集和验证集合
        tdata, vdata = shuffleData(training_data, VALID_RATE)

        # 在训练集上训练
        train_step(tdata, word2ix, tag2ix, model, epoch)

        # 在验证集上验证
        accurary = evaluate(vdata, word2ix, tag2ix, model)

        print '第%d轮，验证集分类准确率：%f' % (epoch + 1, accurary)

        # 如果分类器的准确率在验证集上多次没有提高，就early stop
        if EARLY_STOP:
            # 如果准确率提升，earlystop计数器清零，否则自增
            if accurary >= pre_accurary:
                early_stop_count = 0
                pre_accurary = accurary
            else:
                early_stop_count += 1

            if early_stop_count >= EARLY_STOP_THRESHOLD:
                print 'early stop!!!'
                flag = 'ealystoped'
                break

    # 训练结束在测试集上进行测试
    test_acc = evaluate(test_data, word2ix, tag2ix, model)

    print '测试集准确率 %f' % test_acc
    modelname = 'lstmClassifier-tag1-%s-%d-%f' % (flag, time, test_acc)
    outpath = '../trainedModel/%s.pkl' % modelname
    # 保存模型
    torch.save(model, outpath)
    # 在日志中记录本次训练的模型名称以及准确率
    u.saveModelAcc2txt(modelname, test_acc, LOG_DIR)
