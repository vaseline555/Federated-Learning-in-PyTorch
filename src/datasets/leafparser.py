import os
import PIL
import sys
import json
import torch
import logging
import importlib
import concurrent.futures

from abc import abstractmethod

from src import TqdmToLogger
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
        raise NotImplementedError
        
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return self.num_samples
        
    def __repr__(self):
        return str(self.identifier)

# LEAF - FEMNIST
class FEMNIST(LEAFDataset):
    def __init__(self, in_channels, num_classes, transform=None):
        super(FEMNIST, self).__init__()
        self.in_channels = in_channels
        self.transform = transform
        self.num_classes = num_classes
        
    def _process(self, raw_path):
        inputs = PIL.Image.open(raw_path).convert('L')
        return inputs
    
    def make_dataset(self):
        inputs, targets = self.data['x'], self.data['y']
        self.inputs = [raw_path for raw_path in inputs]
        self.targets = torch.tensor(targets).long()
        self.num_samples = len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = self._process(self.inputs[index]), self.targets[index]
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, targets
    
# LEAF - Shakespeare
class Shakespeare(LEAFDataset):
    def __init__(self, num_embeddings, num_classes):
        super(Shakespeare, self).__init__()
        self.num_embeddings = num_embeddings
        self.num_classes =  num_classes

    def make_dataset(self):
        self.inputs, self.targets = torch.tensor(self.data['x']), torch.tensor(self.data['y'])
        self.num_samples = len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

# LEAF - Sent140
class Sent140(LEAFDataset):
    def __init__(self, num_embeddings, seq_len, num_classes):
        super(Sent140, self).__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len
        self.num_classes = num_classes
        
    def make_dataset(self):
        self.inputs, self.targets = torch.tensor(self.data['x']).long(), torch.tensor(self.data['y']).long()
        self.num_samples = len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

# LEAF - CelebA
class CelebA(LEAFDataset):
    def __init__(self, in_channels, img_path, num_classes, transform=None):
        super(CelebA, self).__init__()
        self.in_channels = in_channels
        self.img_path = img_path
        self.num_classes = num_classes
        self.transform = transform
        
    def _process(self, path):
        inputs = PIL.Image.open(os.path.join(self.img_path, path)).convert('RGB')
        return inputs
    
    def make_dataset(self):
        inputs, targets = self.data['x'], self.data['y']
        self.inputs = [fname for fname in inputs]
        self.targets = torch.tensor(targets).long()
        self.num_samples = len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = self._process(self.inputs[index]), self.targets[index]
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, targets

# LEAF - Reddit
class Reddit(LEAFDataset):
    def __init__(self, num_embeddings, seq_len, num_classes):
        super(Reddit, self).__init__()
        self.num_embeddings = num_embeddings
        self.seq_len = seq_len
        self.num_classes = num_classes

    def make_dataset(self):
        self.inputs, self.targets = torch.tensor(self.data['x']).squeeze(), torch.tensor(self.data['y']).squeeze()
        self.num_samples = len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

def fetch_leaf(args, dataset_name, root, seed, raw_data_fraction, test_size, transforms):
    CONFIG = {
        'femnist': {'in_channels': 1, 'num_classes': 62},
        'shakespeare': {'num_embeddings': 80, 'num_classes': 80},
        'sent140': {'num_embeddings': 400000 + 1, 'seq_len': 25, 'num_classes': 2}, # using GloVe 300-dim embeddings; 400000 + 1 for an unknown token
        'celeba': {'in_channels': 3, 'img_path': f'{root}/celeba/raw/img_align_celeba', 'num_classes': 2},
        'reddit': {'num_embeddings': 10000, 'seq_len': 10, 'num_classes': 10000} # + 1 for an unknown token
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
            tr_dset.identifier = f'[LOAD] [{dataset_name.upper()}] CLIENT < {str(user).zfill(8)} > (train)'
            tr_dset.data = raw_train['user_data'][user]
            tr_dset.make_dataset()
            
            # set essential attributes for test
            te_dset.identifier = f'[LOAD] [{dataset_name.upper()}] CLIENT < {str(user).zfill(8)} > (test)'
            te_dset.data = raw_test['user_data'][user]
            te_dset.make_dataset()
            
            # transplant transform method
            tr_dset.transform = transforms[0]
            te_dset.transform = transforms[1]
            return (tr_dset, te_dset)
        
        datasets = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as workhorse:
            for idx, user in TqdmToLogger(
                enumerate(raw_train['users']), 
                logger=logger, 
                desc=f'[LOAD] [LEAF - {dataset_name.upper()}] ...assigning... ',
                total=len(raw_train['users'])
                ):
                datasets.append(workhorse.submit(_construct_dataset, idx, user).result()) 
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
    args.num_clients = postprocess_leaf(dataset_name.lower(), root, seed, raw_data_fraction=raw_data_fraction, min_samples_per_clients=0, test_size=test_size)
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
    args.num_classes = CONFIG[dataset_name.lower()]['num_classes']
    args.K = len(client_datasets)
    if 'in_channels' in CONFIG[dataset_name.lower()].keys():
        args.in_channels = CONFIG[dataset_name.lower()]['in_channels']
    if 'seq_len' in CONFIG[dataset_name.lower()].keys():
        args.seq_len = CONFIG[dataset_name.lower()]['seq_len']
    if 'num_embeddings' in CONFIG[dataset_name.lower()].keys():
        args.num_embeddings = CONFIG[dataset_name.lower()]['num_embeddings']

    # adjust argument 
    return {}, client_datasets, args
