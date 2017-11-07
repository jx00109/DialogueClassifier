# -*- encoding:utf8 -*-
import numpy as np
import Utils as u
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import xlwt

# **************************************参数设置******************************************** #
TIMES = 10                                                  # 训练的模型个数
datapath = '../data/alldata(onlyEng-fixed12).pkl'           # 数据源位置
tag12path = '../data/tag12.pkl'                             # 一二级字典
dataPath = r'../data/Berlinetta MLK 9.06.xlsx'              # 原始文件
# ***************************************************************************************** #


# 按比例得到训练集、测试集
def divideData(rawdata, data, tag, tag2, trate):
    nsamples = len(data)

    sidx = np.random.permutation(nsamples)
    ntrain = int(np.round(nsamples * (1 - trate)))

    train_data = [data[s] for s in sidx[:ntrain]]
    train_tag = [tag[s] for s in sidx[:ntrain]]
    train_tag2 = [tag2[s] for s in sidx[:ntrain]]
    train_raw = [rawdata[s][0] for s in sidx[:ntrain]]

    test_data = [data[s] for s in sidx[ntrain:]]
    test_tag = [tag[s] for s in sidx[ntrain:]]
    test_tag2 = [tag2[s] for s in sidx[ntrain:]]
    test_raw = [rawdata[s][0] for s in sidx[ntrain:]]

    return train_data, train_tag, train_tag2, train_raw, test_data, test_tag, test_tag2, test_raw


# 得到一级分类每个类别对应的所有小类在二级分类标签数组的索引值
def getLevel2ClassIndex(c1, c2, tag12):
    all_tags = list()
    for each in c1:
        print each
        tags = list()
        for tag in tag12[each]:
            i = np.argwhere(c2 == tag)
            if len(i) != 0:                                 # 处理字典中有的分类在实际样本没有出现的情况
                tags.append(i[0][0])
        all_tags.append(tags)
    return all_tags

# 根据一级分类结果调整二级分类概率分布
def adjustProba(p1,p2,tag22ix):
    for ir, prob1 in enumerate(p1):
        for it, each in enumerate(tag22ix):
            p2[ir][each] = p2[ir][each] * prob1[it]
    return p2

rawData, rawDataRows, rawDataCols = u.openExcel(dataPath=dataPath, index=1)
data = u.loadPickle(datapath)
tag12 = u.loadPickle(tag12path)

rawdata = list()

for i in range(1, rawDataRows):                             # 跳过表头
    r = rawData.row_values(i)
    rawText = r[30]
    text = u.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]            # 获取对话内容以及对应的三级分类
    if u.checkOnlyContainEnglish(text):
        rawdata.append([rawText.strip(), label1.lower(), label2.lower(), label3.lower()])


rawdialogue = list()                                        #原始文本
content = list()                                            #处理后文本
tag = list()                                                #一级标签
tag2 = list()                                               #二级标签

for i, each in enumerate(data):                             # 一级分类样本数5889 二级分类5887
    if each[1].strip() == '' or each[2].strip() == '':
        continue
    else:
        rawdialogue.append(rawdata[i])
        content.append(each[0])
        tag.append(each[1])
        tag2.append(each[2])

for i in range(TIMES):
    train_content, train_tag, train_tag2, train_raw, test_content, test_tag, test_tag2, test_raw = divideData(
        rawdialogue, content, tag, tag2, 0.2)

    v_1 = CountVectorizer(encoding='unicode', stop_words='english')
    t_1 = TfidfTransformer()
    svc_1 = SVC(probability=True, C=0.99, kernel='linear')

    v_2 = CountVectorizer(encoding='unicode', stop_words='english')
    t_2 = TfidfTransformer()
    svc_2 = SVC(probability=True, C=0.99, kernel='linear')

    # 一级分类器
    text_clf_1 = Pipeline(
        [('vect', v_1), ('tfidf', t_1), ('clf', svc_1)])
    # 二级分类器
    text_clf_2 = Pipeline(
        [('vect2', v_2), ('tfidf2', t_2), ('clf2', svc_2)])

    # 独立训练一二级分类器
    text_clf_1 = text_clf_1.fit(train_content, train_tag)
    text_clf_2 = text_clf_2.fit(train_content, train_tag2)

    c1 = text_clf_1.classes_                                # 一级分类类别集合
    c2 = text_clf_2.classes_                                # 二级分类类别集合

    p1 = text_clf_1.predict_proba(test_content)             # 一级分类概率分布
    p2 = text_clf_2.predict_proba(test_content)             # 二级分类概率分布

    tag22ix = getLevel2ClassIndex(c1, c2, tag12)

    # 根据训练好的一级分类器分类结果，将二级分类器分类结果对应大类下的小类概率进行修正
    # 例：
    # 一级分类：A,B,C，预测结果为 0.2， 0.3，0.5
    # 二级分类：a1,a2,b1,b2,c1，二级分类器预测结果为 0.05, 0.15, 0.1, 0.3, 0.4
    # 根据一级分类结果，将二级分类结果修正为 0.05*0.2, 0.15*0.2, 0.1*0.3, 0.3*0.3, 0.4*0.5
    # 修正的方式还需要考虑，该方式结果不好
    p2=adjustProba(p1,p2,tag22ix)
    predix2 = np.argmax(p2, axis=1)
    acc2 = np.mean(c2[predix2] == test_tag2)

    print 'SVM分类器的准确率: %.4f' % acc2

