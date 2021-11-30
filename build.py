import pandas as pd
import numpy as np
import json
import os
import pandas as pd
import random as rd

from tqdm import tqdm


def read_data(file_name):

    data = []
    pbar = tqdm(total=100)
    with open(file_name) as f:
        for l in f:
            seed = rd.random()
            if seed <= 0.3:
                data.append(json.loads(l.rstrip()))
                pbar.update(10)
    pbar.close()
        
    # total length of list, this number equals total number of products

    # first row of the list

    df = pd.DataFrame.from_dict(data)
    df.to_csv('meta.csv', sep = '\t', index = False)

    return df

def read_metadata(file_name):

    data = []
    pbar = tqdm(total=100)
    with open(file_name) as f:
        for l in f:
            data.append(json.loads(l.rstrip()))
            pbar.update(10)
    pbar.close()
        
    # total length of list, this number equals total number of products

    # first row of the list

    df = pd.DataFrame.from_dict(data)
    df.to_csv('meta.csv', sep = '\t', index = False)

    return df

meta_df = read_metadata('./data/meta_Movies_and_TV.json')
print('----------meta_df shape: ' + str(meta_df.shape) + '-------------')
reviews_df = read_data('./data/Movies_and_TV_5.json')
print('----------reviews_df shape: ' + str(reviews_df.shape) + '-------------')
colnames=['asin', 'reviewerID', 'ratings', 'timestamp'] 
ratings_df = pd.read_csv('./data/Movies_and_TV.csv', names = colnames, header = None)[['asin', 'reviewerID', 'ratings']]



reviews_df = reviews_df.merge(ratings_df, on = ['asin', 'reviewerID'], how = 'inner')
reviews_df = reviews_df[~reviews_df.ratings.isna()]
print('----------review after joining ratings shape: ' + str(reviews_df.shape) + '-------------')

reviews_df = reviews_df.merge(meta_df, on = ['asin'], how = 'left')
reviews_df = reviews_df[~reviews_df.category.isna()]
print('----------review after joining meta shape: ' + str(reviews_df.shape) + '-------------')
reviews_df.to_json('all_data.json', orient='records', lines=True)

# reviews_df.to_csv('all_data.csv', sep = '\t', index = False)
