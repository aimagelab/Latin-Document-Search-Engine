# for every folder check the fdimension of the ids and cls embeddings - save the total length in two different lists and print them

import os
import json
import numpy as np
from tqdm import tqdm

path_folder = '/work/pnrr_itserr/WP4-embeddings/index_path/folder_structure'
ids_dimension = []
cls_dimension = []

for el in tqdm(os.listdir(path_folder)):
    folder_data = os.path.join(path_folder, el)
    for e in os.listdir(folder_data):
        if 'tmp_embed_cls' in e:
            embed = np.load(os.path.join(folder_data, e))
            cls_dimension.append(embed.shape[0])
        if 'knn' in e:
            with open(os.path.join(folder_data, e), 'r') as f:
                knn = json.load(f)
            ids_dimension.append(len(knn))
    assert embed.shape[0] == len(knn), f'Error with file: {e}'

print(f'Number: ids {sum(ids_dimension)} - embeds {sum(cls_dimension)}')
print('Done!')

# with open('/work/pnrr_itserr/WP4-embeddings/index_path/folder_structure/job_0010/knn_0010.json', 'r') as f:
#     faiss_list_id = json.load(f)
    
# faiss_list_cls = np.load('/work/pnrr_itserr/WP4-embeddings/index_path/folder_structure/job_0010/tmp_embed_cls_0010.npy')

# len(faiss_list_id)
# faiss_list_cls.shape
