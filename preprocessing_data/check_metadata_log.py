# check the metadata of the job array, it is the first control over the dimension of ids and cls embeddings

import os
import json
import numpy as np
from tqdm import tqdm

count_ok = 0
count_error = 0
ids = []

folder_path = '/work/pnrr_itserr/WP8-embeddings/logs/folder_structure'
for el in tqdm(os.listdir(folder_path)):
    el
    try:
        with open(os.path.join(folder_path, el), "r", encoding="utf-8", errors="replace") as f:
        #with open(os.path.join(folder_path, el), 'r') as f:
            data = f.read()
    except:
        print('Error with file: ', el)
        continue
        
    if 'Correct metadata dimension' in data:
        print('Correct metadata dimension')
        id = data.split('Correct metadata dimension, ')[-1].split('\n')[0]
        ids.append(int(id))
        count_ok += 1
    else:
        count_error += 1
    
print('Number of correct metadata: ', count_ok)
print('Number of error metadata: ', count_error)
print('Done!')
