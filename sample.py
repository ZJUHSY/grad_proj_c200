import json
import pandas as pd
from tqdm import tqdm
import random as rd

data = []
pbar = tqdm(total=100)
with open('all_data.json') as f:
    for l in f:
        seed = rd.random()
        if seed <= 0.05:
            data.append(json.loads(l.rstrip()))
            pbar.update(10)


pbar.close()
drop_cols = ['price', 'image', 'imageURL', 'imageURLHighRes']
df = pd.DataFrame.from_dict(data)
df = df.drop(columns=drop_cols, axis=1)

df.to_json('sample_data.json', orient='records', lines=True)
