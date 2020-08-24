#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: data_loader.py
# @desc:

import pandas as pd

def loadData():
    '''
    载入训练集、测试集数据以及停用词
    :param file_name:
    :return:
    '''
    train_df = pd.read_csv('data/sohu_train.txt', sep='\t', header=None)
    train_df.columns = ['分类', '文章']
    with open('data/stopwords.txt', encoding='utf-8') as file:
        stopword_list = [k.strip() for k in file.readlines() if k.strip() != '']
    print('数据加载成功')
    return train_df, stopword_list

if __name__ == '__main__':
    pass
