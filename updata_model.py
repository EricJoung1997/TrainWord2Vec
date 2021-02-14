import sys
import gensim
import numpy as np
import time
from gensim.models import word2vec
import os
from gensim.models import KeyedVectors

print('加载数据...')
new_sentences = word2vec.Text8Corpus(r'data\updata.txt')

print('加载模型...')
model = gensim.models.Word2Vec.load('model\Library_Word2Vec') 
print('构建词典...')
model.build_vocab(new_sentences, update=True)

time_start=time.time()
print('开始增量训练...')
model.train(new_sentences,epochs=50,total_examples=model.corpus_count)
model.save("literature_AND_libary")

time_end=time.time()

print('增量训练结束！本次训练共：',time_end-time_start,'s') 

