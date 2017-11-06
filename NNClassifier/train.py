# -*- encoding:utf8 -*-
import torch
import torch.autograd as autograd       # torch中自动计算梯度模块
import torch.nn as nn                   # 神经网络模块
import torch.optim as optim             # 模型优化器模块
from lstmModel import LSTMClassifier
import Utils as u
import numpy as np

torch.manual_seed(1)

# ****************************参数设置********************************** #
EMBEDDING_DIM = 100                                     # 词向量维度
HIDDEN_DIM = 100                                        # LSTM隐藏层维度
EPOCH = 50                                              # 训练次数
EARLY_STOP = True                                       # 是否启用early stop
EARLY_STOP_THRESHOLD = 4                                # early stop的阈值
LEARNING_RATE = 0.001                                   # 学习率
VALID_RATE = 0.2                                        # 验证集占比
TEST_RATE = 0.2                                         # 测试集占比
TRAIN_TIMES = 10                                        # 需要的模型总数
LOG_DIR = '../logs/lstm-model-acc.txt'                  # 日志目录
DATA_DIR = '../data/alldata(onlyEng-fixed12).pkl'       # 数据目录
TAG_DIR = '../data/tag12.pkl'                           # 分类标签来源
TAG_LEVEL = 2                                           #分类级别
# ******************************************************************** #

# 按比例得到训练集、验证集、测试集
def divideData(data, vrate, trate):
    nsamples = len(data)

    sidx = np.random.permutation(nsamples)

    nvalid = int(np.round(nsamples * vrate))            # 验证集数据量
    ntest = int(np.round(nsamples * trate))             # 测试集数据量
    ntrain = nsamples - nvalid - ntest                  # 训练集数据量

    train_data = [data[s] for s in sidx[:ntrain]]
    valid_data = [data[s] for s in sidx[ntrain:ntrain + nvalid]]
    test_data = [data[s] for s in sidx[ntrain + nvalid:]]

    return train_data, valid_data, test_data


# 打乱数据集
def shuffleData(data):
    nsamples = len(data)

    sidx = np.random.permutation(nsamples)
    newdata = [data[s] for s in sidx]

    return newdata


# 将原始输入处理成torch接受的格式
def preparexy(seq, word2ix, tag2ix):
    idxs = [word2ix[w] for w in seq[0].split()]
    x = idxs
    y = tag2ix[seq[TAG_LEVEL]]
    return x, y


def getWord2Ix(data):
    word2ix = {}  # 单词的索引字典
    for sent, tag1, _, _ in data:
        for word in sent.split():
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    #加入 #UNK# 用于标记不在词典中的词
    word2ix['#UNK#']=len(word2ix)

    return word2ix


# 该函数只能抽取一级分类
def getTag2Index(tags):
    tag2ix = {}  # 类别的索引字典
    for key in tags:
        if key not in tag2ix:
            tag2ix[key] = len(tag2ix)
    return tag2ix

# 获得二级分类
def getTag2(tags):
    tag2ix={}
    for key in tags:
        for each in tags[key]:
            if each not in tag2ix:
                tag2ix[each]=len(tag2ix)
    return tag2ix

# 计算模型准确率
def evaluate(data, word2ix, tag2ix, model):
    count = .0  # 统计正确分类的样本数
    total = .0  # 统计样本总数
    for i in range(len(data)):
        if data[i][TAG_LEVEL].strip() == '':
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
def train_step(data, word2ix, tag2ix, model, loss_function, optimizer, epoch):
    for i in range(len(data)):
        # 如果没有标签，就直接跳过
        if data[i][TAG_LEVEL].strip() == '':
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


data = u.loadPickle(DATA_DIR)       # 载入数据集
tag12 = u.loadPickle(TAG_DIR)       # 载入分类信息

word2ix = getWord2Ix(data)          # 单词索引字典
if TAG_LEVEL == 1:
    tag2ix = getTag2Index(tag12)    # 一级类别索引字典
else:
    tag2ix = getTag2(tag12)         # 二级类别索引字典

vocab_size = len(word2ix)
tags_size = len(tag2ix)

for time in range(TRAIN_TIMES):
    # 定义模型
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, tags_size)
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义参数优化方法
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_data, valid_data, test_data = divideData(data, VALID_RATE, TEST_RATE)

    early_stop_count = 0            # 统计验证集上准确率连续没有提高的次数
    pre_accurary = .0               # 记录该次训练之前模型最好的准确率

    flag = 'normal'                 # 是否正常完成训练

    for epoch in range(EPOCH):
        # 打乱训练集
        tdata = shuffleData(training_data)

        # 在训练集上训练
        train_step(tdata, word2ix, tag2ix, model, loss_function, optimizer, epoch)

        # 在验证集上验证
        accurary = evaluate(valid_data, word2ix, tag2ix, model)

        print '第%d轮，验证集分类准确率：%.4f' % (epoch + 1, accurary)

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

    print '测试集准确率 %.4f' % test_acc
    modelname = 'lstmClassifier-level%d-%s-%d-%.4f' % (TAG_LEVEL, flag, time, test_acc)
    outpath = '../trainedModel/lstm/%s.pkl' % modelname
    # 保存模型
    torch.save(model, outpath)
    # 在日志中记录本次训练的模型名称以及准确率
    u.saveModelAcc2txt(modelname, test_acc, LOG_DIR)
