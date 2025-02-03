
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import json
from utils.index_utils import extract_texts, extract_sentences_from_texts, encode_sentences
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from transformers import pipeline
import faiss
import pandas as pd
import sqlite3

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])





def main(argv):
    H = FLAGS.config


    #path to json dataset
    json_dataset_path = H.data.json_dataset_path

    #create or open db
    path_to_save_db = os.path.join(H.db.db_path,f"{H.db.db_name}.db")
    connection = sqlite3.connect(path_to_save_db)

    m = connection.total_changes

    assert m == 0, "ERROR: cannot create or open database."

    cursor = connection.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {H.db.db_name} (row_id INTEGER PRIMARY KEY AUTOINCREMENT, author_id TEXT, id TEXT, name TEXT, sentence TEXT, citations TEXT)")
    #cursor.execute(f"ALTER TABLE {H.db.db_name} ADD COLUMN book_name TEXT")

    num_current_folder = 1
    riga_corrente = 0   
    for folder_name in os.listdir(json_dataset_path):
        folder_path = os.path.join(json_dataset_path, folder_name)

        # check if  is a directory
        if os.path.isdir(folder_path):
            print(f"[{num_current_folder}/{len(os.listdir(json_dataset_path))}] Author: {folder_name}")

            # if not (folder_name == "Himerius Soph. (2051)"):
            #      continue

            # Iterate on each json file
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Check if is a JSON file
                if file_name.endswith(".json"):
                    print(f"    JSON: {file_name}")
                    # if not (file_name == "001.json"):
                    #     continue

                    # Leggi il contenuto del file JSON
                    #try:
                    with open(file_path, "r", encoding="utf-8") as json_file:

                        #get json data
                        data = json.load(json_file)

                        # Esegui una query per ottenere tutte le righe
                        cursor.execute(f"SELECT * FROM {H.db.db_name} WHERE row_id >= {riga_corrente}")
                        rows = cursor.fetchall()

                        autore, id, nome = data['author_id'], data['id'], data['name']

                        # Itera sulle righe
                        for row in rows:
                            row_id = row[0]  # Supponiamo che 'row_id' sia la prima colonna (indice 0)
                            riga_corrente = row_id
                            # Condizione per modificare la riga
                            if row[1] == autore and row[2] == id and row[3] == nome:  
                                
                                # Modifica la riga corrente
                                cursor.execute(
                                    f"UPDATE {H.db.db_name} SET book_name = ? WHERE row_id = ?",
                                    (folder_name, row_id),
                                )
                                print(f"    Riga con ID {row_id} modificata.")
                            else:
                                # Se la condizione non è soddisfatta, esci
                                print("     Condizione non soddisfatta. Uscita dal ciclo.\n\n")
                                break
                        connection.commit()
                        #save index
                        #faiss.write_index(index, path_to_save_index)

                    # except Exception as e:
                    #     print(f"    Error opening file {file_name}: {e}")
            num_current_folder+=1
    connection.close()



if __name__ == '__main__':
    app.run(main)