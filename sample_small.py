import json
import pandas as pd
import os
from tqdm import tqdm

data = []
pbar = tqdm(total=100)
with open('sample_data.json') as f:
    cnt = 0
    for l in f:
        cnt += 1
        if cnt >= 50:
            break
        data.append(json.loads(l.rstrip()))
        pbar.update(10)


pbar.close()
df = pd.DataFrame.from_dict(data)
df.to_json('head_data.json', orient='records', lines=True)