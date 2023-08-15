import io
import os
import sys
import csv
import torch
import logging
import torchtext

from src import TqdmToLogger

logger = logging.getLogger(__name__)



# dataset wrapper module
class TextClassificationDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset_name, inputs, targets):
        self.identifier = dataset_name
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self): 
        return len(self.inputs)

    def __getitem__(self, index): 
        inputs = self.inputs[index]
        targets = self.targets[index]
        return inputs, targets
    
    def __repr__(self):
        return str(self.identifier)
    
# helper method to fetch dataset from `torchtext.datasets`
def fetch_torchtext_dataset(args, dataset_name, root, tokenizer, seq_len, num_embeddings):
    URL = {
        'AG_NEWS': 'https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz',
        'SogouNews': 'https://s3.amazonaws.com/fast-ai-nlp/sogou_news_csv.tgz',
        'DBpedia': 'https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz',
        'YelpReviewPolarity': 'https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz',
        'YelpReviewFull': 'https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz',
        'YahooAnswers': 'https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz',
        'AmazonReviewPolarity': 'https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz',
        'AmazonReviewFull': 'https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz'
    }
    MD5 = {
        'AG_NEWS': '2c2d85915f1ca34b29e754ce3b403c81',
        'SogouNews': '45c19a17716907a7a3bde8c7398f7542',
        'DBpedia': '531dab845dd933d0d7b04e82bffe6d96',
        'YelpReviewPolarity': '0f09b3af1a79c136ef9ca5f29df9ed9a',
        'YelpReviewFull': 'a4acce1892d0f927165488f62860cabe',
        'YahooAnswers': '672a634b0a8a1314138321e7d075a64e',
        'AmazonReviewPolarity': 'ee221fbfc3bf7d49dd7dee8064b2404c',
        'AmazonReviewFull': '289723722b64d337a40f809edd29d0f0'
    }
    NUM_CLASSES = {
        'AG_NEWS': 4,
        'SogouNews': 5,
        'DBpedia': 14,
        'YelpReviewPolarity': 2,
        'YelpReviewFull': 5,
        'YahooAnswers': 10,
        'AmazonReviewPolarity': 2,
        'AmazonReviewFull': 5
    }
    
    if dataset_name not in URL.keys():
        err = f'Dataset ({dataset_name}) is not supported!'
        logger.exception(err)
        raise Exception(err)
        
    def _unicode_csv_reader(unicode_csv_data, **kwargs):
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        csv.field_size_limit(maxInt)

        for line in csv.reader(unicode_csv_data, **kwargs):
            yield line
        
    def _csv_iterator(data_path, yield_cls=False):
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        with io.open(data_path, encoding='utf8') as f:
            reader = _unicode_csv_reader(f)
            for row in reader:
                tokens = ' '.join(row[1:])
                tokens = tokenizer(tokens)
                if yield_cls:
                    yield int(row[0]) - 1, torchtext.data.utils.ngrams_iterator(tokens, ngrams=1)
                else:
                    yield torchtext.data.utils.ngrams_iterator(tokens, ngrams=1)

    def _create_data_from_iterator(vocab, iterator, max_len):
        inputs, targets = [], []
        for label, tokens in TqdmToLogger(iterator, logger=logger, desc=f'[LOAD] [{dataset_name.upper()}] ...prepare raw data!'):
            tokens = [vocab[token] for token in tokens]
            # pad tokens to have max length
            pad_len = max_len - len(tokens) % max_len
            if pad_len > 0:
                tokens.extend([vocab['<pad>'] for _ in range(pad_len)])
            
            # slice tokens up to max length
            tokens = tokens[:max_len]

            # collect processed pairs
            inputs.append(tokens)
            targets.append(label)
        return torch.tensor(inputs).long(), torch.tensor(targets).long()
    
    def _create_data_from_tokenizer(tokenizer, iterator, max_len):
        inputs, targets = [], []
        for label, tokens in TqdmToLogger(iterator, logger=logger, desc=f'[LOAD] [{dataset_name.upper()}] ...prepare raw data!'):
            tokens = tokenizer(
                list(tokens),
                return_tensors='pt', 
                is_split_into_words=True,
                max_length=max_len,
                return_attention_mask=False,
                truncation=True,
                padding='max_length'
            )['input_ids']

            inputs.append(*tokens)
            targets.append(label)
        return inputs, targets

    # download files
    logger.info(f'[LOAD] [{dataset_name.upper()}] Start downloading files!')
    root = os.path.expanduser(root)
    raw_files = torchtext.utils.download_from_url(
        url=URL[dataset_name], 
        root=root, 
        hash_value=MD5[dataset_name], 
        hash_type='md5'
    )
    logger.info(f'[LOAD] [{dataset_name.upper()}] ...downloaded files!')
    
    
    # extract archive
    logger.info(f'[LOAD] [{dataset_name.upper()}] Extract archived files!')
    raw_files = torchtext.utils.extract_archive(raw_files)
    logger.info(f'[LOAD] [{dataset_name.upper()}] ...successfully extracted archived files!')
    
    # retrieve split files
    for fname in raw_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname
    
    # build vocabularies using training set
    if tokenizer is None:
        logger.info(f'[LOAD] [{dataset_name.upper()}] Build vocabularies!')
        vocab = torchtext.vocab.build_vocab_from_iterator(_csv_iterator(train_csv_path), specials=['<unk>'], max_tokens=num_embeddings)
        vocab.set_default_index(vocab['<unk>'])
        vocab.vocab.insert_token('<pad>', 0)
        logger.info(f'[LOAD] [{dataset_name.upper()}] ...vocabularies are built!')

    # tokenize training & test data and prepare inputs/targets
    logger.info(f'[LOAD] [{dataset_name.upper()}] Create trainig & test set!')
    if tokenizer is None:
        tr_inputs, tr_targets = _create_data_from_iterator(vocab, _csv_iterator(train_csv_path, yield_cls=True), seq_len)
        te_inputs, te_targets = _create_data_from_iterator(vocab, _csv_iterator(test_csv_path, yield_cls=True), seq_len)
    else:
        tr_inputs, tr_targets = _create_data_from_tokenizer(tokenizer, _csv_iterator(train_csv_path, yield_cls=True), seq_len)
        te_inputs, te_targets = _create_data_from_tokenizer(tokenizer, _csv_iterator(test_csv_path, yield_cls=True), seq_len)

    # adjust labels
    min_label_tr, min_label_te = min(tr_targets), min(te_targets)
    tr_targets = torch.tensor([l - min_label_tr for l in tr_targets]).long()
    te_targets = torch.tensor([l - min_label_te for l in te_targets]).long()
    logger.info(f'[LOAD] [{dataset_name.upper()}] ...created training & test set!')

    # adjust arguments
    args.num_embeddings = len(vocab) + 1 if tokenizer is None else tokenizer.vocab_size
    args.num_classes = NUM_CLASSES[dataset_name]
    return TextClassificationDataset(f'[{dataset_name}] CLIENT', tr_inputs, tr_targets), TextClassificationDataset(f'[{dataset_name}] SERVER', te_inputs, te_targets), args
