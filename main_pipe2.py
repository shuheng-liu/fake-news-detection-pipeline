#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Administrator'
__mtime__ = '2018/9/20'
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
import os
from hyperopt import fmin, hp, tpe, Trials, space_eval
from lightgbm import LGBMClassifier
import numpy as np

loader = EmbeddingLoader("embeddings/")
with open('label.pkl','rb') as f:
    y = pkl.load(f)

name_list =os.listdir(loader.parent_dir)
name_list = [x.replace('.pkl','') for x in name_list if 'onehot' not in x]

for name in name_list:
    X = loader.get_file(loader.parent_dir + name + '.pkl')
    new_model = LearningProcess(X,y)
    space = {
        'num_leaves':  hp.choice('num_leaves',np.arange(10, 200+1, dtype=int)),
        'min_data_in_leaf': hp.choice('min_data_in_leaf',np.arange(10, 200+1, dtype=int)),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.choice('max_bin',np.arange(64, 512, dtype=int)),
        'bagging_freq': hp.choice('bagging_freq',np.arange(1, 5+1, dtype=int)),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10),
        'verbose': -1
    }

    try:
        model, params, train_score, test_score,  train_f1, test_f1 = new_model.try_model_with_hpt(
            model = LGBMClassifier(boosting='gbdt', n_jobs=-1, random_state=2018),
            cv = 5,**space)



    except Exception as e:
        print(e)
        exit()

    with open('model/lgm_from_' + name + "[" + str(train_f1) + ',' + str(test_f1) + "]", 'wb') as f:
        pkl.dump([model, params, train_score, test_score, train_f1, test_f1], f)