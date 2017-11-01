# -*- encoding:utf-8 -*-
import sys
import string
import re

reload(sys)
sys.setdefaultencoding('utf8')

specialsymbols = "[\s+\.\!\/_,$%^*(+\"\'"+string.punctuation+"]+|[+——！，。？<>《》：；、~@#￥%……&*（）]+"
# """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

# 将所有中英文符号替换成空格
def replaceAllSymbols(oldStr):
    return re.sub(specialsymbols.decode("utf8"), " ".decode("utf8"), oldStr)

# 检测是否含有中文
def checkContainChinese(check_str):
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def checkOnlyContainEnglish(check_str):
    for ch in check_str.decode('utf-8'):
        if ch.isalpha() or ch.isdigit() or ch == ' ':
            continue
        else:
            return False
    return True
