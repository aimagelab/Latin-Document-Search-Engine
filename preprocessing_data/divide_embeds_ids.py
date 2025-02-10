# divide in two different folders the ids and cls embeddings

import os
import json
import shutil
from tqdm import tqdm

folder_embeds = '/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/embeds'
folder_ids = '/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/ids'

start_folder = '/work/pnrr_itserr/WP4-embeddings/index_path/folder_structure'

for el in tqdm(os.listdir(start_folder)):
    folder_data = os.path.join(start_folder, el)
    for e in os.listdir(folder_data):
        if 'tmp_embed_cls' in e:
            src = os.path.join(folder_data, e)
            dst = os.path.join(folder_embeds, e)
            shutil.copy(src, dst)
        if 'knn' in e:
            src = os.path.join(folder_data, e)
            dst = os.path.join(folder_ids, e)
            shutil.copy(src, dst)
            
print('Done!')
