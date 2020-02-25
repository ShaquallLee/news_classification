#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: main.py
# @time: 2020/2/25 0025 11:14
# @desc:

from data_loader import loadData
from data_preprocess import sepWords, saveCutwords, loadCutwords
from model_fit import *

if __name__ == '__main__':
    train_df, test_df, stop_words = loadData()
    # cutwords_list = sepWords(train_df, stop_words)
    # print('分词成功')
    # saveCutwords(cutwords_list, 'data/cutwords_list.txt')
    cutwords_list = loadCutwords('data/cutwords_list.txt')
    train_X, test_X, train_y, test_y, label_encoder, tfidf = splitTrainOrTest(train_df, cutwords_list, stop_words, 5, 0.3)
    model = fitModel(train_X, train_y)
    saveModel(model, label_encoder, tfidf)
    score = scoreModel(model, test_X, test_y)
    print("准确率为", score)
