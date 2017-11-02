# -*- encoding:utf8 -*-
'''
该脚本用于获得分类情况，产生两个字典：
每个一级分类对应的子分类
每个二级分类对应的子分类
--------------------------------
输入： *.xlsx
输出： *.pkl (dict)
'''
import Utils as u

dataPath = r'./data/Berlinetta MLK 9.06.xlsx'
outpath12 = r'./data/tag12.pkl'
outpath23 = r'./data/tag23.pkl'

tags12 = dict()  # key: 一级分类 value：一级分类的子分类集合
tags23 = dict()  # key: 二级分类 value：二级分类的子分类集合

data, nrows, ncols = u.openExcel(dataPath=dataPath, index=0)

# 获得类别字典
# key: 分类
# value: 对应分类下的子分类集合
def getClassDict(data, tags, ks, ke, vs, ve):
    for i in range(ks, ke):
        key = data.row_values(0)[i].strip().lower()
        tags[key] = list()
        for j in range(vs, ve):
            lable2 = str(data.row_values(j)[i]).strip().lower()
            if lable2 != '':
                tags[key].append(lable2)
    return tags


tags12 = getClassDict(data, tags12, 0, 5, 1, nrows)
tags23 = getClassDict(data, tags23, 6, ncols, 1, nrows)

u.saveAsPickle(tags12, outpath12)
u.saveAsPickle(tags23, outpath23)
