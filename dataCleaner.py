# -*- encoding:utf8 -*-
'''
该脚本的作用：
1. 去除非英文数据
2. 对原始数据中的标点和不规则单词进行处理
-------------------------------------------
输入: *.xlsx
输出: *.pkl (list)
'''
import Utils

dataPath = r'./data/Berlinetta MLK 9.06.xlsx'
pklDataPath = './data/alldata.pkl'

rawData, rawDataRows, rawDataCols = Utils.openExcel(dataPath=dataPath, index=1)  # 9825 rows, 47 cols

data = list()

for i in range(1, rawDataRows):  # 跳过表头
    r = rawData.row_values(i)
    text = Utils.replaceAllSymbols(r[30])
    label1, label2, label3 = r[42], r[43], r[44]  # 获取对话内容以及对应的三级分类
    if Utils.checkOnlyContainEnglish(text):  # 7173 rows, maxlen=572, minlen=2
        data.append([text.strip().lower(), label1.lower(), label2.lower(), label3.lower()])

# 保存为pkl文件
Utils.saveAsPickle(data, pklDataPath)
