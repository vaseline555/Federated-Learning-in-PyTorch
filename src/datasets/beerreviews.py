import os
import json
import torch
import logging
import torchtext

logger = logging.getLogger(__name__)



class BeerReviews(torch.utils.data.Dataset):
    URL = {
        'look': 'http://people.csail.mit.edu/yujia/files/ls/beer/look.json',
        'aroma': 'http://people.csail.mit.edu/yujia/files/ls/beer/aroma.json'
    } 
    MD5 = {
        'look': '4ad6dd806554ec50ad80b2acedda38d4',
        'aroma': 'd5bc425fb075198a2d4792d33690a3fd'
    } 
    ASPECT = ('look', 'aroma')
    
    def __init__(self, root, aspect, tokenizer=None, download=True):
        assert aspect in self.ASPECT, f'Unknown aspect {aspect}!'
        
        self.root = os.path.expanduser(root)
        self.aspect = aspect
        
        if download:
            self.download()
        if not self._check_exists():
            err = 'Dataset not found or corrupted. You can use download=True to download it'
            logger.exception(err)
            raise RuntimeError(err)

        # get the data and binary targets
        self.inputs, self.targets = self.load_json()

        # Set the maximum sequence length and dataset length
        self.max_seq_len = max([len(text) for text in self.inputs])
        self.length = len(self.targets)

        # get word embeddings from fasttext
        self.emb_bag = self.get_vocab('FastText')

        # set tokenizer
        self.tokenizer = tokenizer
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'beer'))

    def download(self):
        if self._check_exists(): 
            return
        _ = torchtext.utils.download_from_url(
            url=self.URL[self.aspect], 
            root=os.path.join(self.root, 'beer'), 
            hash_value=self.MD5[self.aspect], 
            hash_type='md5'
        )

    def load_json(self):
        inputs, targets = [], []
        path = f'{self.root}/beer/{self.aspect}.json'
        with open(path, 'r') as f:
            for line in f:
                example = json.loads(line)
                targets.append(example['y'])
                inputs.append(example['text'])
        targets = torch.tensor(targets)
        return inputs, targets

    def get_vocab(self, name='FastText'):
        vocab = getattr(torchtext.vocab, name)()

        # Add the pad token
        specials = ['<pad>']
        for token in specials:
            vocab.stoi[token] = len(vocab.itos)
            vocab.itos.append(token)
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, 300)], dim=0)
        return vocab

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        text = self.inputs[index]
        if self.tokenizer is None:
            padded_text = self.emb_bag.stoi['<pad>'] * torch.ones(self.max_seq_len)
            padded_text[:len(text)] = torch.tensor([
                self.emb_bag.stoi[token] if token in self.emb_bag.stoi
                else self.emb_bag.stoi['unk'] for token in text
            ])
            inputs = torch.nn.functional.embedding(padded_text.long(), self.emb_bag.vectors).detach()
        else:
            inputs = self.tokenizer(
                text,
                return_tensors='pt', 
                is_split_into_words=True,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                truncation=True,
                padding='max_length'
            )['input_ids']
        targets = self.targets[index]
        return inputs, targets
    
    def __repr__(self):
        return f'[BeerReviews ({self.aspect})] CLIENT'
    
# helper method to fetch Beer Reviews dataset
def fetch_beerreviews(args, root, aspect='look', tokenizer=None):
    logger.info(f'[LOAD] [BEERREVIEWS] Fetching dataset!')
    
    # create training dataset instance
    raw_train = BeerReviews(root, aspect, tokenizer)
        
    # create test dataset instance
    raw_test = None
    
    logger.info('[LOAD] [BEERREVIEWS] ...fetched dataset!')
    
    # adjust argument
    args.in_features = 300
    args.num_classes = 2
    if tokenizer is None: # use FastText embedding
        args.num_embedings = len(raw_train.emb_bag)
        args.embedding_size = raw_train.emb_bag.dim
    return raw_train, raw_test, args
    