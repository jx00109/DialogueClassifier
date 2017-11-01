# -*- encoding:utf8 -*-
'''
该脚本的作用：
1. 去除非英文数据
2. 对原始数据中的标点和不规则单词进行处理
-------------------------------------------
输入: *.xlsx
输出: *.pkl
'''
import xlrd
import Utils
import re

dataPath = r'./data/Berlinetta MLK 9.06.xlsx'

rawFile = xlrd.open_workbook(dataPath)  # 打开文件
rawData = rawFile.sheets()[1]  # 打开工作表
rawDataRows = rawData.nrows  # 9825
rawDataCols = rawData.ncols  # 47
count = 0
for i in range(1, rawDataRows):
    r = rawData.row_values(i)
    text = Utils.replaceAllSymbols(r[30])
    label1, label2, label3 = r[33], r[34], r[35]  # 获取对话内容以及对应的三级分类
    if Utils.checkOnlyContainEnglish(text): #7244
        count += 1

print count
