import os
import PIL
import sys
import json
import torch
import logging
import importlib

import numpy as np

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from abc import abstractmethod
from multiprocessing import pool

from src.datasets.leaf import *

logger = logging.getLogger(__name__)

    

class LEAFDataset(torch.utils.data.Dataset):
    """Base dataset class for LEAF benchmark dataset.
    """
    def __init__(self):
        super(LEAFDataset, self).__init__()
        self.identifier = None
        self.num_samples = 0
    
    @abstractmethod
    def make_dataset(self):
        err = '[LEAF] A sub-module should implement this method!'
        logger.exception(err)
        raise NotImplementedError(err)
        
    @abstractmethod
    def __getitem__(self, index):
        err = '[LEAF] A sub-module should implement this method!'
        logger.exception(err)
        raise NotImplementedError(err)

    def __len__(self):
        return self.num_samples
    
    def __repr__(self):
        return str(self.identifier)

# LEAF - FEMNIST
class FEMNIST(LEAFDataset):
    def __init__(self, in_channels, transform=None):
        super(FEMNIST, self).__init__()
        self.in_channels = in_channels
        self.transform = transform
        
    def _process(self, raw_path):
        inputs = PIL.Image.open(raw_path).convert('L')
        return inputs
    
    def make_dataset(self):
        inputs, targets = self.data['x'], self.data['y']
        self.inputs = [raw_path for raw_path in inputs]
        self.targets = torch.tensor(targets).long()

    def __getitem__(self, index):
        inputs, targets = self._process(self.inputs[index]), self.targets[index]
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, targets
    
# LEAF - Shakespeare
class Shakespeare(LEAFDataset):
    def __init__(self, num_embeddings, seq_len):
        super(Shakespeare, self).__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len
        
    def make_dataset(self):
        inputs, targets = self.data['x'], self.data['y']
        self.inputs = [word_to_indices(word) for word in inputs]
        self.targets = [letter_to_vec(char) for char in targets]

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]).long(), torch.tensor(self.targets[index]).long()

# LEAF - Sent140
class Sent140(LEAFDataset):
    def __init__(self, num_embeddings, seq_len):
        super(Sent140, self).__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len
        
    def make_dataset(self):
        self.inputs, self.targets = self.data['x'], self.data['y']
        
    def __getitem__(self, index):
        inputs = self.inputs[index]
        targets = self.targets[index]
        return torch.tensor(inputs).long(), torch.tensor(targets).long()

# LEAF - CelebA
class CelebA(LEAFDataset):
    def __init__(self, in_channels, img_path, transform=None):
        super(CelebA, self).__init__()
        self.in_channels = in_channels
        self.img_path = img_path
        self.transform = transform
        
    def _process(self, path):
        inputs = PIL.Image.open(os.path.join(self.img_path, path)).convert('RGB')
        return inputs
    
    def make_dataset(self):
        inputs, targets = self.data['x'], self.data['y']
        self.inputs = [fname for fname in inputs]
        self.targets = torch.tensor(targets).long()
        
    def __getitem__(self, index):
        inputs, targets = self._process(self.inputs[index]), self.targets[index]
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, targets

# LEAF - Reddit
class Reddit(LEAFDataset):
    def __init__(self, num_embeddings, seq_len):
        super(Reddit, self).__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len

    def make_dataset(self):
        self.inputs, self.targets = self.data['x'], self.data['y']
        
    def __getitem__(self, index):
        inputs = self.inputs[index]
        targets = self.targets[index]
        return torch.tensor(inputs).squeeze(), torch.tensor(targets).squeeze()

def fetch_leaf(args, dataset_name, root, seed, raw_data_fraction, test_fraction, n_jobs, transforms):
    NUM_CLASSES = {'femnist': 62, 'shakespeare': 80, 'sent140': 2, 'celeba': 2, 'reddit': 10000}
    CONFIG = {
        'femnist': {'in_channels': 1},
        'shakespeare': {'num_embeddings': 80, 'seq_len': 80},
        'sent140': {'num_embeddings': 400000 + 1, 'seq_len': 25}, # using GloVe 300-dim embeddings; 400000 + 1 for an unknown token
        'celeba': {'in_channels': 3, 'img_path': f'{root}/celeba/raw/img_align_celeba'},
        'reddit': {'num_embeddings': 10001, 'seq_len': 10} # + 1 for an unknown token
    }
        
    def _load_processed(path, mode):
        file = os.listdir(os.path.join(path, mode))[0]
        with open(os.path.join(path, mode, file), 'r') as f:
            proc = json.load(f)
        return proc
    
    def _assign_to_clients(dataset_name, dataset_class, raw_train, raw_test, transforms):
        def _construct_dataset(idx, user):
            # instantiate module for each training set and test set
            tr_dset, te_dset = dataset_class(**CONFIG[dataset_name]), dataset_class(**CONFIG[dataset_name])
            
            # set essential attributes for training
            tr_dset.identifier = f'[{dataset_name.upper()}] CLIENT < {str(user).zfill(8)} > (train)'
            tr_dset.data = raw_train['user_data'][user]
            tr_dset.num_samples = raw_train['num_samples'][idx]
            tr_dset.make_dataset()
            
            # set essential attributes for test
            te_dset.identifier = f'[{dataset_name.upper()}] CLIENT < {str(user).zfill(8)} > (test)'
            te_dset.data = raw_test['user_data'][user]
            te_dset.num_samples = raw_test['num_samples'][idx]
            te_dset.make_dataset()
            
            # transplant transform method
            tr_dset.transform = transforms[0]
            te_dset.transform = transforms[1]
            return (tr_dset, te_dset)
        
        with pool.ThreadPool(processes=n_jobs) as workhorse:
            with logging_redirect_tqdm():
                datasets = workhorse.starmap(_construct_dataset, [(idx, user) for idx, user in tqdm(enumerate(raw_train['users']), leave=False)])
        return datasets
    
    # retrieve appropriate dataset module
    dataset_class = getattr(sys.modules[__name__], dataset_name)
    
    # download data
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(f'{root}/{dataset_name.lower()}/raw'):
        os.makedirs(f'{root}/{dataset_name.lower()}/raw')
        download_data(download_root=f'{root}/{dataset_name.lower()}/raw', dataset_name=dataset_name.lower())
        logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...raw data already exists!')
    
    # pre-process raw data (fetch all raw data into json format)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Pre-process raw data into json format!')
    importlib.import_module(f'.leaf.preprocess.{dataset_name.lower()}', package=__package__).preprocess(root=root)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...done pre-processing raw data into json format!')
    
    # post-process raw data (split data)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Post-process raw data to be split into train & test!')
    args.num_clients = postprocess_leaf(dataset_name.lower(), root, seed, raw_data_fraction=raw_data_fraction, min_samples_per_clients=0, test_data_fraction=test_fraction)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...done post-processing raw data into train & test splits!')
    
    # get raw data
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Load training & test datasets...!')
    raw_train = _load_processed(os.path.join(root, dataset_name.lower()), 'train')
    raw_test = _load_processed(os.path.join(root, dataset_name.lower()), 'test')
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...done parsing trainig & test datasets!')
    
    # make dataset for each client
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Instantiate client datasets and create split hashmap...!')
    client_datasets = _assign_to_clients(dataset_name.lower(), dataset_class, raw_train, raw_test, transforms)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...instantiated client datasets and created split hashmap!')
    
    # adjust arguments
    args.num_classes = NUM_CLASSES[dataset_name.lower()]
    args.K = len(client_datasets)
    if 'in_channels' in CONFIG[dataset_name.lower()].keys():
        args.in_channels = CONFIG[dataset_name.lower()]['in_channels']
    if 'seq_len' in CONFIG[dataset_name.lower()].keys():
        args.seq_len = CONFIG[dataset_name.lower()]['seq_len']
    if 'num_embeddings' in CONFIG[dataset_name.lower()].keys():
        args.num_embeddings = CONFIG[dataset_name.lower()]['num_embeddings']

    # adjust argument 
    return {}, client_datasets, args
