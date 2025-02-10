### List of procedure to create the latin index:
1. launch `preprocessing_data/load_kaggle_file.py` to create the structure and json files for the differnt samples and author books (use .sh file)
2. extract the ids and embeddings (768) using the model `preprocessing_data/extract_index_laberta.py` (use .sh file)
3. control that everythink is ok with the following two scripts: `preprocessing_data/check_metadata_log.py` and `preprocessing_data/check_extracted_data.py`
4. move the file in the appropriate structure to compute the index `preprocessing_data/divide_embeds_ids.py`
5. build the index `scripts/fcocchi/build_index.sh` (use .sh file)
