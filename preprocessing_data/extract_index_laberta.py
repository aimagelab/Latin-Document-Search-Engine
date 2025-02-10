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
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
    print("Using CPU")

# model and tokenizer load
model = AutoModel.from_pretrained(laberta_path, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(laberta_path)
tokenizer.model_max_length = 512
model.eval()

faiss_list_cls = []
faiss_list_id = []
tmp_var = np.zeros((batch_size, 768))
count = 0
count_el = 0

folder_path = '/work/pnrr_itserr/WP4-embeddings/latin_data/db_data'
output_path = '/work/pnrr_itserr/WP4-embeddings/index_path/folder_structure'
# output_path = '/work/pnrr_itserr/WP4-embeddings/index_path/debug'


# TODO: check the name of the output files
TOTAL_PART = int(os.getenv('N_JOBS', 1))
PART = int(os.getenv('JOB', 0))


# if not os.path.exists(tmp_save_directory):
print(f"Creating index: {PART} / {TOTAL_PART}")

tmp_save_dir = os.path.join(output_path, f'job_{str(PART).zfill(4)}')
os.makedirs(tmp_save_dir, exist_ok=True)
tmp_save_directory = os.path.join(tmp_save_dir, f'tmp_embed_cls_{str(PART).zfill(4)}.npy')


debug_for = os.listdir(folder_path) #[:10]
split = len(debug_for) // TOTAL_PART
if PART == TOTAL_PART - 1:
    debug_for = debug_for[split*PART : ]
else:
    debug_for = debug_for[split*PART : (split*PART + split)]


for el_auth in tqdm(debug_for):
    count_el += 1
    print(count_el)
    el_auth_path = os.path.join(folder_path, el_auth)
    for el_sample in os.listdir(el_auth_path):
        el_sample_path = os.path.join(el_auth_path, el_sample)
        with open(el_sample_path, 'r') as f:
            data = json.load(f)
        count_content = 0
        if len(data['content']) > 1:
            for i in range(0, len(data['content']), batch_size):
                            
                with torch.no_grad():
                    val = data['content'][i:i+batch_size]
                    tokenized = tokenizer(val, padding=True, truncation=True, return_tensors='pt').to(model.device)
                    outputs = model(**tokenized)
                    model_cls = outputs.last_hidden_state[:, 0]
                    model_cls = model_cls.cpu().detach().numpy()
                    # model_cls = tmp_var[:len(val)]
                    faiss.normalize_L2(model_cls)
                    try:
                        faiss_list_cls.append(model_cls)
                        count += model_cls.shape[0]

                        for _ in range(0, model_cls.shape[0], 1):
                            count_content += 1
                            faiss_list_id.append(data['id'] + '_' + str(count_content))
                    except:
                        raise Exception (f"Error dimension: {data['id']}")
                    assert count == len(faiss_list_id)
            assert count_content == len(data['content'])

faiss_list_cls = np.concatenate(faiss_list_cls, axis=0)
np.save(tmp_save_directory, faiss_list_cls)

with open(os.path.join(tmp_save_dir, f'knn_{str(PART).zfill(4)}.json'), 'w') as f:
    json.dump(faiss_list_id, f)

# else:
#     print("Loading index")
#     faiss_list_cls = np.load(tmp_save_directory)
# autofaiss.build_index(embeddings=output_path, index_path=output_path+"/knn.index", index_infos_path=output_path+"/index_infos.json")

# final log
print(f"Part: {PART}, len ids {len(faiss_list_id)} and len embeddings {faiss_list_cls.shape[0]}")
if len(faiss_list_id) == faiss_list_cls.shape[0]:
    print(f"Correct metadata dimension, {PART}")
print('Done')
