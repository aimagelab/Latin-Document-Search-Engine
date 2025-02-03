
import numpy as np
import torch
import faiss

def get_best_results(index, H, cursor, query, tokenizer, model, k=1, device=None):

    encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)

    # Tokenize sentence
    inputs = tokenizer(query, return_tensors="pt").to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)

    encoding[0] = sentence_embedding

    # Normalize
    faiss.normalize_L2(encoding)

    #search best matches in index
    distances, ann = index.search(encoding, k=k)


    json_result = {}

    r=0
    #retrieve best results from db
    for idx in ann[0]:
        cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
        rows = cursor.fetchall()
        json_result[r] = {}
        for row in rows:
            json_result[r]['author_id'] = row[1]
            json_result[r]['id'] = row[2]
            json_result[r]['name'] = row[3]
            json_result[r]['sentence'] = row[4]
            json_result[r]['citations'] = row[5]
            json_result[r]['book_name'] = row[6]
        r+=1
    
    return json_result