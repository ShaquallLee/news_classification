#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: shuffle_split.py
# @time: 2020/2/25 0025 15:07
# @desc:

import pandas as pd

from data_loader import loadData
from model_fit import loadModel

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score

if __name__ == '__main__':
    train_df, test_df, stop_words = loadData()
    tfidf_vectorizer, label_encoder, model = loadModel('data/tfidf.model')
    X = tfidf_vectorizer.transform(train_df['文章'])
    y = label_encoder.transform(train_df['分类'])

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    cv_split = ShuffleSplit(n_splits=5, test_size=0.3)
    score_ndarray = cross_val_score(model, X, y, cv=cv_split)
    print(score_ndarray)
    print(score_ndarray.mean())
