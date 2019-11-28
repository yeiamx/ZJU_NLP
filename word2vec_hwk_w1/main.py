# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import pandas as pd
from model import *

# gensim version
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# sentences = word2vec.Text8Corpus("data/text8.txt")  # 加载语料
# model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5

# 计算两个词的相似度/相关程度
# y1 = model.similarity("woman", "man")
# print (u"woman和man的相似度为：", y1)
# print ("--------\n")

# self made version
model = SelfWord2Vec('data/text8.txt') #建立并训练模型

#训练完成后可以这样调用
# model = Word2Vec() #建立空模型
# model.load_model('myvec') #从当前目录下的myvec文件夹加载模型
# pd.Series(model.most_similar('fruit'))