# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# **************************************参数设置********************************************* #
TIMES = 10                                                  #训练的模型个数
datapath = '../data/alldata(fixed12).pkl'                #数据源位置
TAG_LEVEL = 2                                               #分类级别
# ******************************************************************************************* #


# 按比例得到训练集、测试集
def divideData(data, tag, trate):
    nsamples = len(data)

    sidx = np.random.permutation(nsamples)
    ntrain = int(np.round(nsamples * (1 - trate)))

    train_data = [data[s] for s in sidx[:ntrain]]
    train_tag = [tag[s] for s in sidx[:ntrain]]
    test_data = [data[s] for s in sidx[ntrain:]]
    test_tag = [tag[s] for s in sidx[ntrain:]]

    return train_data, train_tag, test_data, test_tag

data = u.loadPickle(datapath)
content = [d[0] for d in data]      # 文本
tag1 = [d[1] for d in data]         # 一级分类
tag2 = [d[2] for d in data]         # 二级分类
tag3 = [d[3] for d in data]         # 三级分类

tags = [tag1, tag2, tag3]
for i in range(TIMES):
    tag = tags[TAG_LEVEL - 1]
    train_content, train_tag, test_content, test_tag = divideData(content, tag, 0.2)

    vectorizer = CountVectorizer(encoding='unicode', stop_words='english')
    tfidftransformer = TfidfTransformer()
    # tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  # 先转换成词频矩阵，再计算TFIDF值
    # print tfidf.shape #(5738, 23439)

    text_clf = Pipeline([('vect', vectorizer), ('tfidf', tfidftransformer), ('clf', SVC(C=0.99, kernel='linear'))])
    text_clf = text_clf.fit(train_content, train_tag)
    predicted = text_clf.predict(test_content)
    acc=np.round(np.mean(predicted == test_tag), 4)
    print 'SVM分类器的准确率: %.4f' % acc
    modelname='svm-level%d-%d-%.4f' % (TAG_LEVEL, i, acc)
    u.saveAsPickle(text_clf,'../trainedModel/svm/%s.pkl' % modelname)
    u.saveModelAcc2txt(modelname, acc, '../logs/svm-model-acc.txt')
    # 随机选取测试集中的一条数据进行分类
    ix = np.random.randint(1, 100)
    # print ix
    # print test_content[ix]
    print test_tag[ix]
    print text_clf.predict([test_content[ix]])
