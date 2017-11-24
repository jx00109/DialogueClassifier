# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from collections import Counter
import xlwt

# **************************************参数设置******************************************** #
TIMES = 1  # 训练的模型个数
datapath = '../data/alldata(onlyEng-fixed12).pkl'  # 数据源位置
TAG_LEVEL = 1  # 分类级别(5 60)
N_FEATURES = 500
N_TOPICS = 5
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

for i in range(TIMES):
    train_content, train_tag, train_raw, test_content, test_tag, test_raw = divideData(rawdialogue, content, tag, 0.2)
    # 得到单词-文档共现矩阵
    vectorizer = CountVectorizer(encoding='unicode', stop_words='english', max_features=N_FEATURES)

    train_data = vectorizer.fit_transform(train_content)

    train_tag = np.array(train_tag)

    test_data = vectorizer.fit_transform(test_content)  # [n_samples, n_features]

    model = LDA(n_topics=N_TOPICS, max_iter=5, batch_size=128)
    model.fit(train_data)

    train_data_distr = model.transform(train_data)
    pred_tag = train_data_distr.argmax(axis=1)

    # 投票
    id2class = dict()
    for idx in range(N_TOPICS):
        idxs = np.where(pred_tag == idx)[0]
        # print Counter(train_tag[idxs])
        id2class[idx] = Counter(train_tag[idxs]).most_common(1)[0][0]
    print id2class
    doc_topic_distr = model.transform(test_data)  # [n_samples, n_topics]
    class_id = doc_topic_distr.argmax(axis=1)
    pred = [id2class[each] for each in class_id]
    pred=np.array(pred)
    test_tag=np.array(test_tag)
    acc=np.mean(pred==test_tag)
    print 'LDA分类器的准确率: %.4f' % acc
    """
    modelname = 'lda-level%d-%d-%.4f' % (TAG_LEVEL, i, acc)
    u.saveAsPickle(text_clf, '../trainedModel/lda/%s.pkl' % modelname)
    u.saveModelAcc2txt(modelname, acc, '../logs/lda-model-acc.txt')

    outpath = '../results/lda/%s.xls' % modelname
    workbook = xlwt.Workbook(encoding='utf8')
    worksheet = workbook.add_sheet('实验结果')

    for i, each in enumerate(text_clf.predict(test_content)):
        worksheet.write(i, 0, test_raw[i])              # 原始文本
        worksheet.write(i, 1, test_content[i])          # 处理过的文本
        worksheet.write(i, 2, test_tag[i])              # 原始标签
        worksheet.write(i, 3, each)                     # 预测标签

    workbook.save(outpath)
    """
