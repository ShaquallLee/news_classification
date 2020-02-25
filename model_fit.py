#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: model_fit.py
# @time: 2020/2/25 0025 10:59
# @desc:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

def splitTrainOrTest(train_df, cutwords_list, stopwords, min_df, max_df):
    '''
    编码并切分训练集测试集
    :param train_df:
    :param cutwords_list:
    :param stopwords:
    :param min_df:
    :param max_df:
    :return:
    '''
    tfidf = TfidfVectorizer(cutwords_list, stop_words=stopwords, min_df=min_df, max_df=max_df)
    X = tfidf.fit_transform(train_df["文章"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df["分类"])
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    return train_X, test_X, train_y, test_y, label_encoder, tfidf


def fitModel(train_X, train_y):
    '''
    训练模型
    :param train_X:
    :param train_y:
    :return:
    '''
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(train_X, train_y)
    return model

def scoreModel(model, test_X, test_y):
    '''
    评价模型
    :param model:
    :param test_X:
    :param test_y:
    :return:
    '''
    return model.score(test_X, test_y)

def saveModel(model, labelEncoder, tfidf):
    '''
    保存模型
    :param model:
    :param labelEncoder:
    :param tfidf:
    :return:
    '''
    try:
        with open('data/tfidf.model', 'wb') as file:
            save = {
                'labelEncoder': labelEncoder,
                'tfidfVectorize': tfidf,
                'logistic_model': model
            }
            pickle.dump(save, file)
            print('模型保存成功：data/tfidf.model')
    except:
        print("模型保存失败")

def loadModel(file_path):
    '''
    加载模型
    :param file_path:
    :return:
    '''
    with open(file_path, 'rb') as file:
        model_mes = pickle.load(file)
        tfidf_vectorizer = model_mes['tfidfVectorizer']
        label_encoder = model_mes['labelEncoder']
        model = model_mes['logistic_model']
        return tfidf_vectorizer, label_encoder, model

if __name__ == '__main__':
    pass
