# -*- encoding:utf-8 -*-
import sys
import string
import re
import xlrd
import pickle as pkl

reload(sys)
sys.setdefaultencoding('utf8')

specialsymbols = "[\s+\.\!\/_,$%^*(+\"\'" + string.punctuation + "]+|[+——！，。？<>《》：；、~@#￥%……&*（）]+"
mathsysmbols = '\d+(\.\d+)*([×\+\-\*\/]\d+(\.\d+)*)*[0-9A-Za-z]*'


# """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

# 将所有中英文符号替换成空格
def replaceAllSymbols(oldStr):
    # 去掉数字
    oldStr = re.sub(mathsysmbols.decode("utf-8"), " ".decode("utf-8"), oldStr)
    # 再去掉符号
    return re.sub(specialsymbols.decode("utf8"), " ".decode("utf8"), oldStr)


# 检测是否含有中文
def checkContainChinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def checkOnlyContainEnglish(check_str):
    if check_str.strip() == '':
        return False
    for ch in check_str.decode('utf-8'):
        if ch.isalpha() or ch.isdigit() or ch == ' ':
            continue
        else:
            return False
    return True


# 打开excel文件
def openExcel(dataPath, index):
    rawFile = xlrd.open_workbook(dataPath)  # 打开文件
    rawData = rawFile.sheets()[index]  # 打开工作表
    rawDataRows = rawData.nrows  # 9825
    rawDataCols = rawData.ncols  # 47
    return rawData, rawDataRows, rawDataCols


def saveAsPickle(data, outpath):
    with open(outpath, 'wb') as f:
        pkl.dump(data, f)


def loadPickle(datapath):
    with open(datapath, 'rb') as f:
        return pkl.load(f)
