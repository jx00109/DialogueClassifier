# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import xlwt

# **************************************参数设置******************************************** #
TIMES = 10
datapath = '../data/alldata(onlyEng-fixed12).pkl'
TAG_LEVEL = 2
# ***************************************************************************************** #

dataPath = r'../data/Berlinetta MLK 9.06.xlsx'

rawData, rawDataRows, rawDataCols = u.openExcel(dataPath=dataPath, index=1)

rawdata = list()

for i in range(1, rawDataRows):
    r = rawData.row_values(i)
    rawText = r[30]
    text = u.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]
    if u.checkOnlyContainEnglish(text):
        rawdata.append([rawText.strip(), label1.lower(), label2.lower(), label3.lower()])


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


def vote(mixRes):
    finalRes = list()
    for i in range(len(mixRes[0])):
        res = list()
        for j in range(len(mixRes)):
            res.append(mixRes[j][i])
        temp = Counter(res).most_common(1)[0]
        name = temp[0]
        num = temp[1]
        if num > len(mixRes) / 2:
            finalRes.append(name)
        else:
            finalRes.append(res[0])
    finalRes = np.array(finalRes)
    return finalRes


data = u.loadPickle(datapath)
rawdialogue = list()
content = list()
tag = list()
for i, each in enumerate(data):
    if each[TAG_LEVEL].strip() == '':
        continue
    else:
        rawdialogue.append(rawdata[i])
        content.append(each[0])
        tag.append(each[TAG_LEVEL])
print 'num of classes: %d' % len(dict(Counter(tag)))
total_acc = 0

for time in range(TIMES):
    train_content, train_tag, train_raw, test_content, test_tag, test_raw = divideData(rawdialogue, content, tag, 0.2)

    vectorizer = CountVectorizer(encoding='unicode', stop_words='english')
    tfidftransformer = TfidfTransformer()

    svm_clf = Pipeline([('vect', vectorizer), ('tfidf', tfidftransformer), ('clf', SVC(C=0.99, kernel='linear'))])
    svm_clf = svm_clf.fit(train_content, train_tag)
    svm_pred = svm_clf.predict(test_content)
    svm_acc = np.mean(svm_pred == test_tag)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn_clf = Pipeline([('vect', vectorizer), ('tfidf', tfidftransformer), ('clf', knn)])
    knn_clf = knn_clf.fit(train_content, train_tag)
    knn_pred = knn_clf.predict(test_content)
    knn_acc = np.mean(knn_pred == test_tag)

    bayes_clf = Pipeline([('vect', vectorizer), ('tfidf', tfidftransformer), ('clf', MultinomialNB())])
    bayes_clf = bayes_clf.fit(train_content, train_tag)
    bayes_pred = bayes_clf.predict(test_content)
    bayes_acc = np.mean(bayes_pred == test_tag)

    final_pred = vote([svm_pred, knn_pred, bayes_pred])
    final_acc = np.mean(final_pred == test_tag)
    total_acc += final_acc

    modelname = 'mix-level%d-%d-%.4f' % (TAG_LEVEL, time, final_acc)
    u.saveAsPickle(svm_clf, '../trainedModel/mix/svm-%s.pkl' % modelname)
    u.saveAsPickle(knn_clf, '../trainedModel/mix/knn-%s.pkl' % modelname)
    u.saveAsPickle(bayes_clf, '../trainedModel/mix/bayes-%s.pkl' % modelname)
    u.saveModelAcc2txt(modelname, final_acc, '../logs/mix-model-acc.txt')

    outpath = '../results/mix/%s.xls' % modelname
    workbook = xlwt.Workbook(encoding='utf8')
    worksheet = workbook.add_sheet('实验结果')

    for i, each in enumerate(final_pred):
        worksheet.write(i, 0, test_raw[i])
        worksheet.write(i, 1, test_content[i])
        worksheet.write(i, 2, test_tag[i])
        worksheet.write(i, 3, each)

    workbook.save(outpath)

print 'average acc: %.4f' % (total_acc / TIMES)
