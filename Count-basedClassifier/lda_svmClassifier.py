# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.svm import SVC

# **************************************参数设置******************************************** #
TIMES = 10  # 训练的模型个数
datapath = '../data/alldata(onlyEng-fixed12).pkl'  # 数据源位置
TAG_LEVEL = 1  # 分类级别(5 60)
N_FEATURES = 500
N_TOPICS = 100
# ***************************************************************************************** #

dataPath = r'../data/Berlinetta MLK 9.06.xlsx'

rawData, rawDataRows, rawDataCols = u.openExcel(dataPath=dataPath, index=1)  # 9825 rows, 47 cols

rawdata = list()

for i in range(1, rawDataRows):  # 跳过表头
    r = rawData.row_values(i)
    rawText = r[30]
    text = u.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]  # 获取对话内容以及对应的三级分类
    if u.checkOnlyContainEnglish(text):  # 6152 rows, maxlen=572, minlen=2
        rawdata.append([rawText.strip(), label1.lower(), label2.lower(), label3.lower()])

dataPath = r'../data/Berlinetta MLK 9.06.xlsx'
outpath = '../results/bestResult-%d.xls' % TAG_LEVEL

rawData, rawDataRows, rawDataCols = u.openExcel(dataPath=dataPath, index=1)  # 9825 rows, 47 cols

rawdata = list()

for i in range(1, rawDataRows):  # 跳过表头
    r = rawData.row_values(i)
    rawText = r[30]
    text = u.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]  # 获取对话内容以及对应的三级分类
    if u.checkOnlyContainEnglish(text):  # 7173 rows, maxlen=572, minlen=2
        rawdata.append([rawText.strip(), label1.lower(), label2.lower(), label3.lower()])


# 按比例得到训练集、测试集
def divideData(rawdata, data, tag, trate):
    nsamples = len(data)

    sidx = np.random.permutation(nsamples)
    ntrain = int(np.round(nsamples * (1 - trate)))

    train_data = [data[s] for s in sidx[:ntrain]]
    train_tag = [tag[s] for s in sidx[:ntrain]]
    train_raw = [rawdata[s][0] for s in sidx[:ntrain]]
    test_data = [data[s] for s in sidx[ntrain:]]
    test_tag = [tag[s] for s in sidx[ntrain:]]
    test_raw = [rawdata[s][0] for s in sidx[ntrain:]]

    return train_data, train_tag, train_raw, test_data, test_tag, test_raw


data = u.loadPickle(datapath)

rawdialogue = list()
content = list()
tag = list()

for i, each in enumerate(data):  # 一级分类样本数5889 二级分类5887
    if each[TAG_LEVEL].strip() == '':
        continue
    else:
        rawdialogue.append(rawdata[i])
        content.append(each[0])
        tag.append(each[TAG_LEVEL])
total_acc = 0
for i in range(TIMES):
    train_content, train_tag, train_raw, test_content, test_tag, test_raw = divideData(rawdialogue, content, tag, 0.2)
    # 得到单词-文档共现矩阵
    vectorizer = CountVectorizer(encoding='unicode', stop_words='english', max_features=N_FEATURES)

    train_data = vectorizer.fit_transform(train_content)
    test_data = vectorizer.fit_transform(test_content)  # [n_samples, n_features]

    model = LDA(n_topics=N_TOPICS, batch_size=64)
    model.fit(train_data)

    dt_matrix = model.transform(train_data)
    test_dt_matrix = model.transform(test_data)
    svc = SVC(C=0.99, kernel='linear')

    svc = svc.fit(dt_matrix, train_tag)
    pred = svc.predict(test_dt_matrix)
    acc = np.round(np.mean(pred == test_tag), 4)
    total_acc += acc
    print 'LDA分类器的准确率: %.4f' % acc
print 'average accuary: ', total_acc / TIMES
