import pandas as pd
import numpy as np
import json
import os
import re
from typing import Callable

from bs4 import BeautifulSoup
import re
import nltk.data
import nltk
from nltk.corpus import stopwords


from zipfile import ZipFile
# zf = ZipFile('Dataset_B.zip', 'r')
# zf.extractall('.')
import random as rd
pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
nltk.download('punkt')



def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, 'html.parser').get_text()
    review_text = re.sub('[^a-zA-z]', ' ', review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        words = [w for w in words if not w in stops]
    return words

# 加载nltk的tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# 将评论分解成已解析的句子,返回句子列表
# 每个句子都是有单词构成的列表
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # 使用NLTK分词器，把段子分成句子
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        # 如果句子是空串，跳过不处理
        if len(raw_sentence) > 0:
        # 调用上面的方法得到单词列表
            sentences.append(review_to_wordlist(raw_sentence,
                                   remove_stopwords))
    return sentences


data = []
with open('sample_data.json') as f:
    cnt = 0
    for l in f:
        # cnt += 1
        # if cnt >= 50:
        #     break
        data.append(json.loads(l.rstrip()))


df = pd.DataFrame.from_dict(data)
# df.to_json('head_data.json', orient='records', lines=True)
model_cols = ['reviewerID', 'asin', 'reviewText', 'ratings']
model_df = df[model_cols]
model_df.reviewText = model_df.reviewText.fillna('')
model_df['Sentences'] = model_df['reviewText'].apply(lambda x : review_to_sentences(x, tokenizer))



msk = np.random.rand(len(df)) < 0.8
train_df = model_df[msk]
test_df = model_df[~msk]

if not os.path.isdir('./model/'):
    os.mkdir('./model/')
train_df.to_json('./model/train_data.json', orient='records', lines=True)
test_df.to_json('./model/test_data.json', orient='records', lines=True)
