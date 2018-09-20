#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Administrator'
__mtime__ = '2018/9/19'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from embedding_loader import EmbeddingLoader
import pandas as pd
import pickle as pkl
from class_demo import LearningProcess
from hyperopt import fmin, hp, tpe, Trials, space_eval
from sklearn.naive_bayes import MultinomialNB

loader = EmbeddingLoader("embeddings/")
with open('label.pkl','rb') as f:
    y = pkl.load(f)

name_list = [
        'text-onehot(scorer=count)',
        'text-onehot(scorer=count, normalized)',
        'text-onehot(scorer=tfidf)',
        'text-onehot(scorer=tfidf, normalized)',
        'title-onehot(scorer=count)',
        'title-onehot(scorer=count, normalized)',
        'title-onehot(scorer=tfidf)',
        'title-onehot(scorer=tfidf, normalized)'
]
for name in name_list:
    X = loader.get_file(loader.parent_dir + name + '.pkl')

    new_model = LearningProcess(X,y)


    space ={
            "alpha":hp.loguniform("alpha",-4, 1)
           }

    model, params, train_score, test_score,  train_f1, test_f1 = new_model.try_model_with_hpt(model = MultinomialNB(),cv = 5,**space)

    with open('model/nb_from_' + name+"["+str(train_f1)+','+str(test_f1)+"]", 'wb') as f:
         pkl.dump([model,params,train_score, test_score,  train_f1, test_f1], f)