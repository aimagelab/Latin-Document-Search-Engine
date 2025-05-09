from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference

def get_config():
    config = ConfigDict()


    config.run = run = ConfigDict()

    #gpu device
    run.device = 2

    config.data = data = ConfigDict()

    '''
        The dataset is in this format:
            - dataset_name
                - author_1
                    - <...>.json
                    - <...>.json
                    ...
                    - <...>.json
                - author_2
                    - <...>.json
                    - <...>.json
                    ...
                    - <...>.json
                ...
        
        So, json_dataset_path is the absolute path to 'dataset_name'
    '''
    data.json_dataset_path = "/work/pnrr_itserr/WP4-embeddings/latin_data/db_data"

    '''
    When creating the index, sentences are broken up into low and high points. However, some sentences, if broken up, do not make much sense on their own.

    e.g. ...τῆς βασιλείας αὐτοῦ νῦν καὶ εἰς τοὺς αἰῶνας τῶν αἰώνων. ἀμήν.

    It is a prayer that ends with "amen". It would be ideal to keep it in the sentence.
    The min_words_in_phrase parameter indicates how many words at least a sentence must have to be "alone", separate from the others.
    '''
    data.min_words_in_phrase = 5

    #lenght of the embedding of each sentence. Will be used to inizialize the index
    data.len_embedding = 768
    data.window_data = 1


    config.model = model = ConfigDict()

    model.tokenizer = "/work/pnrr_itserr/WP8-embeddings/checkpoints/models--bowphs--LaBerta/snapshots/94fab85783dca8a16529cda2b58760d03bd5d9c1"
    model.model = "/work/pnrr_itserr/WP8-embeddings/checkpoints/models--bowphs--LaBerta/snapshots/94fab85783dca8a16529cda2b58760d03bd5d9c1"
    model.model_max_length = 512 #max number of tokens the model can handle
    #when decide to join two words w1-w2, check if w2 is in the top_k next words after w1
    model.top_k = 30
    model.top_author = 1000
    model.filter_max_depth = 1000000

    config.index = index = ConfigDict()
    
    index.index_path = "/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/embeds"
    index.idx_2_keys = "/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/ids/knn.json"
    index.index_name ="knn.index"
    
    # debug mode
    # index.index_path = "/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/debug"
    # index.idx_2_keys = "/work/pnrr_itserr/WP4-embeddings/index_path/division_folder/debug/knn.json"

    config.db = db = ConfigDict()
    db.db_path = ""
    db.db_name ="DB_Latin"


    #not important. Just if you want to execute test_index.py
    config.retrieval = retrieval = ConfigDict()
    retrieval.num_matches = 5

    return config
