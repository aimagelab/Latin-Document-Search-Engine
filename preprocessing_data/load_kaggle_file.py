# import kagglehub
# path = kagglehub.dataset_download("yaustal/latin-literature-dataset-170m")
# print("Path to dataset files:", path)

import os
import re
import json
from tqdm import tqdm 
import pandas as pd
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from preprocessing_data.contractions import contractions, tokenize_clean

# path_lemmas = '/work/pnrr_itserr/WP4-embeddings/latin_data/3/latin_lemmas.csv'
# df_lemmas = pd.read_csv(path_lemmas, nrows=1000)

db_data = '/work/pnrr_itserr/WP4-embeddings/latin_data/db_data'

path_raw = '/work/pnrr_itserr/WP4-embeddings/latin_data/3/latin_raw.csv'
df_raw = pd.read_csv(path_raw, nrows=1000) #, nrows=1000

df_raw.head()
sent_tokenizer = SentenceTokenizer()
check_data_id = []

for author, samples in tqdm(df_raw.groupby('author'), mininterval=1, maxinterval=df_raw.shape[0]):
    author_formatted = author.strip().replace(' ', '_')
    folder_author_name = os.path.join(db_data, author_formatted)
    os.makedirs(folder_author_name, exist_ok=True)
    for row_index, sample in samples.iterrows():
        data = {}
        data['author'] = sample['author']
        data['title'] = sample['title']
        # data['text'] = sample['text']
        id = sample['Unnamed: 0'].split('\\')[-1].split('.')[0]
        data['id'] = author_formatted + '_' + id
        check_data_id.append(data['id'])
        
        clean_text = tokenize_clean(sample['text'])
        clean_text = sent_tokenizer.tokenize(clean_text)
        if len(clean_text) == 1:
            data['content'] = clean_text
        else:
            data['content'] = clean_text[:len(clean_text)//2]
        
        with open(os.path.join(folder_author_name, data['id'] + '.json'), 'w') as f:
            json.dump(data, f)

print(len(check_data_id))
print(len(set(check_data_id)))

print('Done')
