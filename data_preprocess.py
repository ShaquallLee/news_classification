#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: data_preprocess.py
# @desc:

import jieba

def sepWords(df, stopwords):
    '''
    分词
    :param df:
    :param stopwords:
    :return:
    '''
    cutwords_list = []
    for article in df['文章']:
        cutwords = [k for k in jieba.lcut(article) if k not in stopwords]
        cutwords_list.append(cutwords)
        print('--', article)
    return cutwords_list

def saveCutwords(cutwords_list, file_name):
    '''
    保存分词结果
    :param cutwords_list:
    :return:
    '''
    with open(file_name, 'w+', encoding='utf-8') as file:
        for cutwords in cutwords_list:
            file.write(' '.join(cutwords)+'\n')

def loadCutwords(file_name):
    '''
    加载分词结果
    :param file_name:
    :return:
    '''
    with open(file_name, 'r', encoding='utf-8') as file:
        cutwords_list = [k.strip().split() for k in file.readlines()]
    print('分词结果加载成功')
    return cutwords_list


if __name__ == '__main__':
    pass
