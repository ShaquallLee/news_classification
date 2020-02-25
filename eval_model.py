#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: eval_model.py
# @time: 2020/2/25 0025 15:32
# @desc:

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from model_fit import loadModel


def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]

if __name__ == '__main__':
    test_df = pd.read_csv('data/sohu_test.txt', sep='\t', header=None)
    tfidf_vectorizer, label_encoder, model = loadModel('data/tfidf.model')
    test_X = tfidf_vectorizer.transform(test_df[1])
    test_y = label_encoder.transform(test_df[0])
    predict_y = model.predict(test_X)
    res = eval_model(test_y, predict_y, label_encoder.classes_)
    print(res)
