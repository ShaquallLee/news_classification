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
    print('加载数据成功')
    # cutwords_list = sepWords(train_df, stop_words)
    # print('分词成功')
    # saveCutwords(cutwords_list, 'data/cutwords_list.txt')
    cutwords_list = loadCutwords('data/cutwords_list.txt')
    print('加载分词结果成功')
    train_X, test_X, train_y, test_y = splitTrainOrTest(train_df, cutwords_list, stop_words, 40, 0.3)
    print('切分成功')
    model = fitModel(train_X, train_y)
    print('得到模型')
    score = scoreModel(model, test_X, test_y)
    print('得到评分')
    print(score)
