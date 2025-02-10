#  file to build the index

import os
import json
from tqdm import tqdm
from natsort import natsorted
import autofaiss

path_ids = '/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/ids'
data_ids = []

shards = natsorted(os.listdir(path_ids))
for shard in tqdm(shards):
    if 'knn.json' not in shard:
        shard_path = os.path.join(path_ids, shard)
        with open(shard_path, 'r') as f:
            data = json.load(f)
        data_ids.extend(data) # popolate the data_ids list with data
        
path_tot_ids = os.path.join(path_ids, 'knn.json')
json.dump(data_ids, open(path_tot_ids, 'w'))
print('Done ids!')

output_path = '/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/embeds'
autofaiss.build_index(embeddings=output_path, index_path=output_path+"/knn.index", index_infos_path=output_path+"/index_infos.json")

print('Done embeds!')
print('Done!')
