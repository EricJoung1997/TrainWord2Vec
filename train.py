import sys
import gensim
# import sklearn
import numpy as np
import jieba
import time
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.models import word2vec
from jieba import analyse
TaggededDocument = gensim.models.doc2vec.TaggedDocument
import os


def main():
    
    num_features = 100    # 将词汇映射到的N维空间的维度数量（N）
    min_word_count = 1   # 用于修剪内部字典
    num_workers = 16       # 用于训练并行化的参数，可以加快训练速度
    context = 5          # 当前词和预测词之间的最大距离
    iter  =  10       #模型训练时在整个训练语料库上的迭代次数
    downsampling = 1e-3   # 高频词汇的随机降采样的配置阈值
    # sg 模型所采用的算法类型，1 代表 skip-gram，0代表 CBOW
    sentences = word2vec.Text8Corpus('TrainWord2Vec\data\output_data.txt')
    # more_sentences = word2vec.Text8Corpus("Dir.txt")
    model = word2vec.Word2Vec(sentences, workers=num_workers,iter=iter, \
            size=num_features, min_count = min_word_count, \
            window = context, sg = 1, sample = downsampling)
    model.init_sims(replace=True)
    # 保存模型
    model.save("TrainWord2Vec\model\Word2VecModel_LIS")
    
    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model = gensim.models.Word2Vec.load('modelKG')
    # model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)

if __name__ == "__main__":
    time_start=time.time()
    main()
    time_end=time.time()
    print('本次训练共：',time_end-time_start,'s') 