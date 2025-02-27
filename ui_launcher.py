import gradio as gr
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import os
import json
from utils.launcher_utils import get_best_results, custom_get_best_results, custom_get_best_results_filtered
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
import faiss
import pandas as pd
import numpy as np

# Commandline arguments
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/index_config.py", "configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

import unicodedata
import re

from bs4 import BeautifulSoup

def highlight_words_in_html(html, parole, stile):
    soup = BeautifulSoup(html, "html.parser")  
    
    for parola in parole:
        # Trova e sostituisci solo il testo (non i tag)
        for elem in soup.find_all(string=re.compile(r'\b' + re.escape(parola) + r'\b')):
            nuovo_contenuto = re.sub(
                r'\b' + re.escape(parola) + r'\b', 
                f'<span style="{stile}">{parola}</span>', 
                elem
            )
            elem.replace_with(BeautifulSoup(nuovo_contenuto, "html.parser"))
    
    return str(soup)

def normalize_text(text):
    # Rimuovere caratteri invisibili come spazi non separabili
    text = text.replace('\xa0', ' ')  # Sostituire \xa0 con uno spazio
    # Normalizzazione Unicode
    text = unicodedata.normalize('NFKC', text)
    # Rimuovere eventuali spazi extra all'inizio e alla fine
    text = text.strip()
    # Rimuovere caratteri invisibili come nuove righe e tabulazioni
    text = re.sub(r'\s+', ' ', text)  # Sostituire sequenze di spazi con un singolo spazio
    return text

def main(argv):

    H = FLAGS.config
    
    # create a list of authors, used to filter the results
    list_of_works = [e for e in os.listdir(H.data.json_dataset_path)]
    list_of_works = list(set(list_of_works))
    list_of_works.insert(0, 'All')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(H.model.tokenizer)
    model = AutoModel.from_pretrained(H.model.model).to(device)
    tokenizer.model_max_length = 512
    model.eval()
    # model = AutoModelForMaskedLM.from_pretrained(H.model.model).to(device)

    # load index and keys
    index = faiss.read_index(os.path.join(H.index.index_path,f"{H.index.index_name}"))
    with open(H.index.idx_2_keys, 'r') as f:
        idx_2_keys = json.load(f)
    

    # def process_inputs(text, number):
    def process_inputs(text, number, works_selected, additional_text, additional_text_slider_value):

        if text=="":
            return f"""
            <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; background-color: white;">
                <p>Insert a valid query.</p>
            </div> """
        
        number = H.model.top_k
        warning_author = None
        if works_selected is None:
            best_results = custom_get_best_results(index, H, idx_2_keys, text, tokenizer, model, number, device, additional_text, additional_text_slider_value)
        else:
            max_depth = H.model.filter_max_depth
            best_results, warning_author = custom_get_best_results_filtered(index, H, idx_2_keys, text, tokenizer, model, number, device, additional_text, additional_text_slider_value, works_selected, max_depth)

        # Creare una stringa HTML per visualizzare tutti i risultati
        results_html = ""
        if len(best_results) == 0:
            results_html += f"""
            <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; color: white; background-color: #1f2937">
                <strong>Attention: No results found for {works_selected} in the first {max_depth} items.</strong>
            </div>
            """
            return results_html
        
        best_results = dict(list(best_results.items())[:number])
        len_best_results = len(list(best_results.items()))
        
        results = []
        for i in range(len_best_results):
            result = {
                "Book": best_results[i]['book_name'],
                "id": best_results[i]['id'],
                "name": best_results[i]['name'],
                "context": best_results[i]['context'],
                "exact_match": best_results[i]['exact_match']
            }
            results.append(result)
            
        if warning_author == True:
            results_html += f"""
            <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; color: white; background-color: #1f2937">
                <strong>Attention: The following sentences may not be semantically related.</strong>
            </div>
            """
            
            for result in results:
                match_text = result['context'] # window of context of the passage
                cit = result['exact_match'] # represent the exact retrieved passage
            
                # Creare una box per ogni risultato
                results_html += f"""
                <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; color: white; background-color: #1f2937">
                    <strong>Book:</strong> {result['Book']}<br>
                    <strong>id:</strong> {result['id']}<br>
                    <strong>name:</strong> {result['name']}<br>
                    <strong>match:</strong> {match_text}<br>
                    <strong>citation:</strong> {cit}<br>
                </div>
                """
        
        else:
            for result in results:
                match_text = result['context'] # window of context of the passage
                cit = result['exact_match'] # represent the exact retrieved passage
            
                # Creare una box per ogni risultato
                results_html += f"""
                <div style="border: 2px solid #ccc; padding: 10px; margin-bottom: 10px; color: white; background-color: #1f2937">
                    <strong>Book:</strong> {result['Book']}<br>
                    <strong>id:</strong> {result['id']}<br>
                    <strong>name:</strong> {result['name']}<br>
                    <strong>match:</strong> {match_text}<br>
                    <strong>citation:</strong> {cit}<br>
                </div>
                """
                # <strong>author_id:</strong> {result['author_id']}<br>

        results_html += """
        <style>
            .citation {
                color: rgb(1, 3, 39);  /* Text color */
                text-decoration-line: underline;  /* Underline text */
                background-color: #f8f9fa;  /* Change background color */
                padding: 2px 5px; /* Optional: Add padding for better visibility */
                border-radius: 3px; /* Optional: Round the edges */
            }

            /* Aggiungere un effetto al passaggio del mouse */
            .citation:hover {
                
                font-weight: bold;  /* Opzionale: rendere il testo più evidente */
                background-color: #e6e6e6;  /* Un piccolo effetto di sfondo per l'hover */
            }

            /* Aggiungi un effetto di tooltip che appare al passaggio del mouse */
            .citation[data-citation]:hover::after {
                content: attr(data-citation);  /* Mostra il contenuto del tooltip */
                position: absolute;
                background: rgba(0, 0, 109, 0.8);
                color: white;
                padding: 5px;
                border-radius: 5px;
                font-size: 12px;
                white-space: nowrap;
                z-index: 9999;
            }
                    </style>
        """
        
        return results_html


    demo = gr.Interface(
        fn=process_inputs,
        inputs=[
            gr.Textbox(lines=5, 
                        placeholder="Servius ad Virgil. Aen. III, 334: [Chaonios cognomine Campos] Epirum campos non habere omnibus notum est.",
                        label="Enter query"),
            gr.Dropdown(["DB_Latin"], label="Select the database where to search"),
            gr.Dropdown(choices=list_of_works, label="Select Author", multiselect=False),
            gr.Textbox(label="Additional Phrase", placeholder="Enter an additional phrase"),
            gr.Slider(0, 1, step=0.01, label="How much weight should be given to the additional sentence compared to the main one.")
        ],
        outputs=gr.HTML(),
        title="Latin Document Search Engine",
        description="Enter a text query and the number of results you want to get. The system will search the documents for the best results and automatically sort them.",
    )


    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == '__main__':
    app.run(main)
    