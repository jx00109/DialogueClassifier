# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import xlwt

# **************************************参数设置******************************************** #
TIMES = 10  # 训练的模型个数
datapath = '../data/alldata(onlyEng-fixed12).pkl'  # 数据源位置
TAG_LEVEL = 2  # 分类级别
# ***************************************************************************************** #

dataPath = r'../data/Berlinetta MLK 9.06.xlsx'

rawData, rawDataRows, rawDataCols = u.openExcel(dataPath=dataPath, index=1)

rawdata = list()

for i in range(1, rawDataRows):  # 跳过表头
    r = rawData.row_values(i)
    rawText = r[30]
    text = u.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]  # 获取对话内容以及对应的三级分类
    if u.checkOnlyContainEnglish(text):  # 6152 rows, maxlen=572, minlen=2
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

print len(rawdialogue)
print len(content)
print len(tag)
for i in range(TIMES):
    train_content, train_tag, train_raw, test_content, test_tag, test_raw = divideData(rawdialogue, content, tag, 0.2)

    vectorizer = CountVectorizer(encoding='unicode', stop_words='english')
    tfidftransformer = TfidfTransformer()
    knn = KNeighborsClassifier(n_neighbors=5)

    text_clf = Pipeline([('vect', vectorizer), ('tfidf', tfidftransformer), ('clf', knn)])
    text_clf = text_clf.fit(train_content, train_tag)
    pred = text_clf.predict(test_content)
    acc = np.round(np.mean(pred == test_tag), 4)
    # p= text_clf.predict_proba(test_content)
    # predicted=np.argmax(p,axis=1)
    # acc = np.round(np.mean(text_clf.classes_[predicted] == test_tag), 4)
    print 'KNN分类器的准确率: %.4f' % acc
    '''
    modelname = 'svm-level%d-%d-%.4f' % (TAG_LEVEL, i, acc)
    u.saveAsPickle(text_clf, '../trainedModel/knn/%s.pkl' % modelname)
    u.saveModelAcc2txt(modelname, acc, '../logs/knn-model-acc.txt')

    
    outpath = '../results/svm/%s.xls' % modelname
    workbook = xlwt.Workbook(encoding='utf8')
    worksheet = workbook.add_sheet('实验结果')

    for i, each in enumerate(text_clf.predict(test_content)):
        worksheet.write(i, 0, test_raw[i])                      # 原始文本
        worksheet.write(i, 1, test_content[i])                  # 处理过的文本
        worksheet.write(i, 2, test_tag[i])                      # 原始标签
        worksheet.write(i, 3, each)                             # 预测标签

    workbook.save(outpath)
    '''
