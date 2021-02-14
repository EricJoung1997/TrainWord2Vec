import sys
import numpy as np
import time
import gensim
from gensim.models import word2vec
import jieba
from jieba import analyse
import jieba.analyse
import re
from string import digits
import pandas as pd
from tqdm import tqdm
import os


model =  gensim.models.Word2Vec.load('model\Library_Word2Vec')

print(model.wv.most_similar('学习共享空间',topn=10))