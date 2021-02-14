import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
import time
import pandas as pd
import numpy as np
import os
import tqdm
mp=os.getcwd()

Dir=mp+'\data\Dir.txt'
Fpath=mp+'\data\StopWords.txt'


time_start=time.time()

jieba.load_userdict(Dir)
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(Fpath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    sentence1 = sentence.lower()#大写转小写
    sentence_seged = jieba.cut(sentence1.strip())
    stopwords = stopwordslist(Fpath)  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

count=0
inputs = open('data\input_data.txt', 'r', encoding='utf-8')
outputs = open('data\output_data.txt', 'w', encoding='utf-8')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
    count += 1
    print("第{}行分词成功！".format(count))
outputs.close()
inputs.close()

time_end=time.time()
print('本次分词共：',time_end-time_start,'s')