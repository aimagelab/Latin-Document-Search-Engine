import numpy as np
import json

with open('/work/pnrr_itserr/WP4-embeddings/index_path/complete/knn.json', 'r') as f:
    faiss_list_id = json.load(f)
    
faiss_list_cls = np.load('/work/pnrr_itserr/WP4-embeddings/index_path/tmp_embed_cls.npy')

len(faiss_list_id)
faiss_list_cls.shape
