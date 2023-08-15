import os
import json
import logging

from collections import Counter, defaultdict

logger = logging.getLogger(__name__)



def preprocess(root):
    def _make_path(path, dir_name):
        if not os.path.exists(os.path.join(path, dir_name)):
            os.makedirs(os.path.join(path, dir_name))
        
    def _load_data(path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data
    
    def _save_data(path, data):
        with open(path, 'w') as file:
            json.dump(data, file)
    
    def _refine_data(data):
        num_samples = []
        for user in data['users']:
            num_samples.append(len(data['user_data'][user]['x'])) # get correct sample counts
            data['user_data'][user]['y'] = [original if (type(original) is list) else original['target_tokens'] for original in data['user_data'][user]['y']] # don't know why... but some samples are not parsed (i.e., in `dict` format, not `list`)
        else:
            data['num_samples'] = num_samples
        return data
    
    def _build_counter(train_data):
        all_tokens = []
        for u in train_data:
            for c in train_data[u]['x']:
                for s in c:
                    all_tokens.extend(s)    
        counter = Counter()
        counter.update(all_tokens)
        return counter

    def _build_vocab(counter):
        vocab_size = 10000
        pad_symbol, unk_symbol, bos_symbol, eos_symbol = 0, 1, 2, 3
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        count_pairs = count_pairs[:vocab_size - 1]
        
        words, _ = list(zip(*count_pairs))
        words = list(words)
        vocab = {}
        vocab['<PAD>'] = pad_symbol
        vocab['<UNK>'] = unk_symbol
        vocab['<BOS>'] = bos_symbol
        vocab['<EOS>'] = eos_symbol

        idx = 4 # due to special tokens
        while len(words) > 0:
            w = words.pop()
            if w in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                continue
            vocab[w] = idx
            idx += 1
        vocab = {'vocab': vocab, 'size': vocab_size, 'unk_symbol': unk_symbol, 'pad_symbol': pad_symbol, 'bos_symbol': bos_symbol, 'eos_symbol': eos_symbol}
        return vocab
    
    def _convert_to_ids_and_get_length(raw, vocab):
        def _tokens_to_word_ids(tokens, vocab):
            return [vocab[word] for word in tokens]
        
        def _convert_to_id(container, key):
            transformed = []
            for data in container[key]:
                for sent in data:
                    idx = _tokens_to_word_ids(sent, vocab)
                    transformed.append([idx])
            return transformed
        
        for user in raw['users']:
            raw['user_data'][user]['x'] = _convert_to_id(raw['user_data'][user], 'x')
            raw['user_data'][user]['y'] = _convert_to_id(raw['user_data'][user], 'y')
        return raw
    
    # set path
    DATASET_NAME = __file__.split('/')[-1].split('.')[0]
    path = os.path.join(os.path.expanduser(root), DATASET_NAME)
    
    # check if preprocessing has already done
    _make_path(path, 'all_data')
    _make_path(path, 'vocab')
    _make_path(path, 'intermediate')
    _make_path(path, 'train')
    _make_path(path, 'test')
        
    # adjust path since preprocessed json files are already prepared
    if os.path.exists(os.path.join(path, 'raw', 'new_small_data')):
        for file in os.listdir(os.path.join(path, 'raw', 'new_small_data')):
            if 'train' in file:
                os.replace(os.path.join(path, 'raw', 'new_small_data', file), os.path.join(path, 'intermediate', file))
            elif 'test' in file:
                os.replace(os.path.join(path, 'raw', 'new_small_data', file), os.path.join(path, 'intermediate', file))
            else:
                os.remove(os.path.join(path, 'raw', 'new_small_data', file)) # `val` data is not required...
    
    # edit `num_samples`: don't know why but it is not correct...
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Refine raw data...!')
    train_data = _load_data(os.path.join(path, 'intermediate', 'train_data.json'))
    test_data = _load_data(os.path.join(path, 'intermediate', 'test_data.json'))
    
    # correct number of samples and filter tokenized samples only
    train_data = _refine_data(train_data)
    test_data = _refine_data(test_data)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished refining raw data!')
    
    # aggreagte data
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Combine raw data...!')
    _save_data(os.path.join(path, 'all_data', 'train_data_refined.json'), train_data)
    _save_data(os.path.join(path, 'all_data', 'test_data_refined.json'), test_data)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...finished combining raw data!')
    
    # build vocabulary
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Build vocabulary...!')
    counter = _build_counter(train_data['user_data'])
    vocab_raw = _build_vocab(counter)
    _save_data(os.path.join(path, 'vocab', 'reddit_vocab.json'), vocab_raw)
    
    vocab = defaultdict(lambda: vocab_raw['unk_symbol'])
    vocab.update(vocab_raw['vocab'])
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...vocabulary is successfully created!')
    
    # convert tokens to index
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Convert tokens into indices using vocabulary...!')
    train_data = _convert_to_ids_and_get_length(train_data, vocab)
    test_data = _convert_to_ids_and_get_length(test_data, vocab)    
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...all tokens are converted into indices!')
    
    # save processed data
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] Split into training & test sets...!')
    _save_data(os.path.join(path, 'train', 'all_data_niid_00_train.json'), train_data)
    _save_data(os.path.join(path, 'test', 'all_data_niid_00_test.json'), test_data)
    logger.info(f'[LOAD] [LEAF - {DATASET_NAME.upper()}] ...done splitting into training & test sets!')
    