import os
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import autofaiss
from autofaiss import build_index
from tqdm import tqdm

batch_size = 32
laberta_path = '/work/pnrr_itserr/WP8-embeddings/checkpoints/models--bowphs--LaBerta/snapshots/94fab85783dca8a16529cda2b58760d03bd5d9c1'

# model and tokenizer load
model = AutoModel.from_pretrained(laberta_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(laberta_path)
tokenizer.model_max_length = 512
model.eval()

faiss_list_cls = []
faiss_list_id = []

folder_path = '/work/pnrr_itserr/WP4-embeddings/latin_data/db_data'
output_path = '/work/pnrr_itserr/WP4-embeddings/index_path'
# output_path = '/work/pnrr_itserr/WP4-embeddings/index_path/debug'

debug_for = os.listdir(folder_path) #[:10]
for el_auth in tqdm(debug_for, mininterval=1, maxinterval=len(os.listdir(folder_path))):
    el_auth_path = os.path.join(folder_path, el_auth)
    for el_sample in os.listdir(el_auth_path):
        el_sample_path = os.path.join(el_auth_path, el_sample)
        with open(el_sample_path, 'r') as f:
            data = json.load(f)
        for i in tqdm(range(0, len(data['content']), batch_size)):
                        
            with torch.no_grad():
                val = data['content'][i:i+batch_size]
                tokenized = tokenizer(val, padding=True, truncation=True, return_tensors='pt').to(model.device)
                outputs = model(**tokenized)
                model_cls = outputs.last_hidden_state[:, 0]
                model_cls = model_cls.cpu().detach().numpy()
                faiss.normalize_L2(model_cls)
                faiss_list_cls.append(model_cls)
                
            for j in range(0, len(val), 1):
                faiss_list_id.append(data['id'] + '_' + str(j))

faiss_list_cls = np.concatenate(faiss_list_cls, axis=0)

tmp_save_directory = os.path.join(output_path, 'tmp_embed_cls.npy')
np.save(tmp_save_directory, faiss_list_cls)
autofaiss.build_index(embeddings=output_path, index_path=output_path+"/knn.index", index_infos_path=output_path+"/index_infos.json")

with open(os.path.join(output_path, 'knn.json'), 'w') as f:
    json.dump(faiss_list_id, f)
    
print('Done')
