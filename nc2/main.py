#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: main.py
# @time: 2020/8/24 0024 18:05
# @desc:

import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data_news = pd.read_table('./data/data.txt',names=['category','theme','URL','content'],encoding='utf-8')
stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
print('数据加载成功')

data_news = data_news.dropna()
content = data_news.content.values.tolist()
content_split = []
for line in content:
    seg = jieba.lcut(line)
    reseg = []
    for word in seg:
        if word not in stopwords:
            reseg.append(word)
    if len(seg)>1 and seg != '\r\n':
        content_split.append(' '.join(reseg))
print('分词成功')

vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train = pd.DataFrame({'content': content_split,
                         'label':list(map(lambda x:label_mapping[x], data_news['category']))})

x_train, x_test, y_train, y_test = train_test_split(df_train['content'].values, df_train['label'].values, random_state=999)

model = MultinomialNB()

model.fit(vec.fit_transform(x_train), y_train)
print('模型训练成功')

score = model.score(vec.transform(x_test), y_test)
print('评分为', score)