import os
import re
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)



def preprocess(root):
    RAW_TRAINING = 'training.1600000.processed.noemoticon.csv'
    RAW_TEST  = 'testdata.manual.2009.06.14.csv'
    
    def _get_glove_vocab(path):
        # read GloVe embeddings (300-dim)
        lines = []
        with open(os.path.join(path, 'raw', 'glove.6B.300d.txt'), 'r') as file:
            lines = file.readlines()
        
        # process embeddings
        lines = [l.split() for l in lines]
        vocab = [l[0] for l in lines]
        return vocab
    
    def _combine_data(path):
        raw_train = pd.read_csv(
            os.path.join(path, RAW_TRAINING), 
            encoding='ISO-8859-1', 
            header=None, 
            names=['target', 'id', 'date', 'flag', 'user', 'text'],
            usecols=['target', 'user', 'text'], 
            index_col='user'
        )
        raw_test = pd.read_csv(
            os.path.join(path, RAW_TEST), 
            encoding='ISO-8859-1', 
            header=None, 
            names=['target', 'id', 'date', 'flag', 'user', 'text'],
            usecols=['target', 'user', 'text'], 
            index_col='user'
        )
        raw_all = pd.concat([raw_train, raw_test]).sort_index(kind='mergesort')
        return raw_all
        
    def _convert_to_json(path, raw_all, vocab):
        def _get_words_indices(vocab):
            """Get word indices using pre-defined GloVe. 
            """
            indices = {}
            for i in range(len(vocab)):
                indices[vocab[i]] = i
            vocab = {w: i for i, w in enumerate(vocab)}
            return indices
        
        def _split_line(line):
            """Split given line/phrase into list of words
            """
            return re.findall(r"[\w']+|[.,!?;]", line)

        def _line_to_indices(line, word2id, max_words=25):
            """Converts given phrase into list of word indices.

            - If the phrase has more than `max_words` words, 
            returns a list containing indices of the first `max_words` words.

            - If the phrase has less than `max_words` words, 
            repeatedly appends integer representing padding index to returned list 
            until the list's length is `max_words`.

            Args:
                line: string representing phrase/sequence of words
                word2id: dictionary with string words as keys and int indices as values
                max_words: maximum number of word indices in returned list

            Returns:
                indices: list of word indices, one index for each word in phrase
            """
            pad_id = 0
            line_list = _split_line(line) # split phrase in words
            indices = [word2id[w] if w in word2id else pad_id for w in line_list[:max_words]]
            indices += [pad_id] * (max_words - len(indices))
            return indices

        # get index dictionry of each word
        indices = _get_words_indices(vocab)

        # get required elements
        num_samples = raw_all.groupby('user').apply(len)
        users = list(num_samples.to_dict().keys())
        raw_all['target'] = raw_all['target'].replace({4: 1})
        raw_all['text'] = raw_all['text'].apply(lambda x: _line_to_indices(x, indices))
        user_data = raw_all.reset_index().groupby('user').agg({'text': lambda x: [i for i in x], 'target': lambda y: [l for l in y]}).rename(columns={'text': 'x', 'target': 'y'}).T.to_dict()
        
        # create json file
        all_data = {}
        all_data['users'] = users
        all_data['num_samples'] = num_samples.values.tolist()
        all_data['user_data'] = user_data
        
        # save file
        with open(os.path.join(path, 'all_data', 'all_data.json'), 'w') as outfile:
            json.dump(all_data, outfile)

    # set path
    DATASET_NAME = __file__.split('/')[-1].split('.')[0]
    path = os.path.join(os.path.expanduser(root), DATASET_NAME)

    # check if preprocessing has already done
    if not os.path.exists(os.path.join(path, 'all_data')):
        os.makedirs(os.path.join(path, 'all_data'))
    else:
        return
    
    # make path for GloVe vocabulary
    if not os.path.exists(os.path.join(path, 'vocab')):
        os.makedirs(os.path.join(path, 'vocab'))
    
    # get GloVe vocabulary
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Process GloVe embeddings (300 dim.)...!')
    glove_vocab = _get_glove_vocab(path)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished processing GloVe embeddings!')
    
    # combine raw data
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Combine raw data...!')
    raw_all = _combine_data(os.path.join(path, 'raw'))
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished combining raw data!')
    
    # convert to json format 
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Convert data to json format... (this may take several minutes)!')
    _convert_to_json(path, raw_all, glove_vocab)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished converting data to json format!')
