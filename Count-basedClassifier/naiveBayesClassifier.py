# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import xlwt

# **************************************参数设置********************************************* #
TIMES = 10                                              # 训练的模型个数
datapath = '../data/alldata(onlyEng-fixed12).pkl'       # 数据源位置
TAG_LEVEL = 1                                           # 分类级别
# ******************************************************************************************* #

dataPath = r'../data/Berlinetta MLK 9.06.xlsx'


rawData, rawDataRows, rawDataCols = u.openExcel(dataPath=dataPath, index=1)  # 9825 rows, 47 cols

rawdata = list()

for i in range(1, rawDataRows):                     # 跳过表头
    r = rawData.row_values(i)
    rawText = r[30]
    text = u.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]    # 获取对话内容以及对应的三级分类
    if u.checkOnlyContainEnglish(text):             # 6152 rows, maxlen=572, minlen=2
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
for i, each in enumerate(data):                     # 5889 5887
    if each[TAG_LEVEL].strip() == '':
        continue
    else:
        rawdialogue.append(rawdata[i])
        content.append(each[0])
        tag.append(each[TAG_LEVEL])

for i in range(TIMES):
    train_content, train_tag, train_raw, test_content, test_tag, test_raw = divideData(rawdialogue, content, tag, 0.2)

    vectorizer = CountVectorizer(encoding='unicode', stop_words='english')
    tfidftransformer = TfidfTransformer()

    text_clf = Pipeline([('vect', vectorizer), ('tfidf', tfidftransformer), ('clf', MultinomialNB())])
    text_clf = text_clf.fit(train_content, train_tag)
    predicted = text_clf.predict(test_content)
    acc = np.round(np.mean(predicted == test_tag), 4)
    print 'Bayes分类器的准确率: %.4f' % acc
    modelname = 'bayes-level%d-%d-%.4f' % (TAG_LEVEL, i, acc)
    u.saveAsPickle(text_clf, '../trainedModel/bayes/%s.pkl' % modelname)
    u.saveModelAcc2txt(modelname, acc, '../logs/bayes-model-acc.txt')

    outpath = '../results/bayes/%s.xls' % modelname
    workbook = xlwt.Workbook(encoding='utf8')
    worksheet = workbook.add_sheet('实验结果')

    for i, each in enumerate(text_clf.predict(test_content)):
        worksheet.write(i, 0, test_raw[i])          # 原始文本
        worksheet.write(i, 1, test_content[i])      # 处理过的文本
        worksheet.write(i, 2, test_tag[i])          # 原始标签
        worksheet.write(i, 3, each)                 # 预测标签

    workbook.save(outpath)
