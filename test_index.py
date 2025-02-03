'''
    Example how to get similar vectors from index giving a phrase
'''


from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import json
from utils.index_utils import extract_texts, extract_sentences_from_texts, encode_sentences
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3
import numpy as np


# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])



def main(argv):

    H = FLAGS.config

    device = H.run.device if torch.cuda.is_available() else -1

    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)
    mask_filler = pipeline("fill-mask", model = H.model.model, tokenizer = H.model.tokenizer, top_k = H.model.top_k, device=device)

    #load index
    index = faiss.read_index(os.path.join(H.index.index_path,f"{H.index.index_name}.index"))

    #create or open db
    path_to_load_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_load_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot open database."

    cursor = connection.cursor()


    text = 'μιμησάσθω καὶ λόγους, εἰ δυνατόν, ἡ γραφὴ διὰ προσώπου στυγνάζοντος, καὶ πᾶς τις τῶν ῥημά των ἀκουσάτω διὰ τοῦ σχήματος.' #"προσαγορεύομεν τοὺς ἐν Χαδουθὶ πάντας κατ' ὄνομα." #  # "προσαγορεύομεν τοίνυν πάντες ἡμεῖς οἱ τεσσαράκοντα ἀδελφοὶ καὶ συνδέσμιοι πάντες Μελέτιος Ἀέτιος Εὐτύχιος Κυρίων Κάνδιδος Ἀγγίας Γάϊος Χουδίων Ἡράκλειος Ἰωάννης Θεόφιλος Σισίνιος Σμάραγδος Φιλοκτήμων Γοργόνιος Κύριλλος Σεβηριανὸς Θεόδουλος Νίκαλλος Φλάβιος Ξάνθιος Οὐαλέριος Ἡσύχιος Δομετιανὸς Δόμνος Ἡλιανὸς Λεόντιος ὁ καὶ Θεόκτιστος Εὐνοϊκὸς Οὐάλης Ἀκάκιος Ἀλέξανδρος Βικράτιος ὁ καὶ Βιβιανὸς Πρίσκος Σακέρδων Ἐκδίκιος Ἀθανάσιος Λυσίμαχος Κλαύδιος Ἴλης καὶ Μελίτων."

    encoding = np.zeros((1, H.data.len_embedding)).astype(np.float32)

    # Tokenize sentence
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use [CLS] token embedding as sentence encoding
    sentence_embedding = outputs['hidden_states'][-1][:,0,:].squeeze().cpu().numpy().astype(np.float32)

    encoding[0] = sentence_embedding

    # Normalize
    faiss.normalize_L2(encoding)

    #search best matches in index
    distances, ann = index.search(encoding, k=H.retrieval.num_matches)

    #retrieve best results from db
    for idx in ann[0]:
        cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id = {idx+1}")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
            print("")




    print("")


if __name__ == '__main__':
    app.run(main)