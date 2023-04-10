import os
import torch
import logging
import torchtext
import torchvision
import transformers

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from collections import ChainMap
from multiprocessing import pool

from src.datasets import *
from src.loaders.split import simulate_split

logger = logging.getLogger(__name__)

    

class SubsetWrapper(torch.utils.data.Dataset):
    """Wrapper of `torch.utils.data.Subset` module for applying individual transform.
    """
    def __init__(self, subset, suffix):
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)
    
    def __repr__(self):
        return f'{repr(self.subset.dataset.dataset)} {self.suffix}'

def load_dataset(args):
    """Fetch and split requested datasets.
    
    Args:
        args: arguments
        
    Returns:
        split_map: {client ID: [assigned sample indices]}
            ex) {0: [indices_1], 1: [indices_2], ... , K: [indices_K]}
        server_testset: (optional) holdout dataset located at the central server, 
        client datasets: [(local training set, local test set)]
            ex) [tuple(local_training_set[indices_1], local_test_set[indices_1]), tuple(local_training_set[indices_2], local_test_set[indices_2]), ...]

    """
    TOKENIZER_STRINGS = {
        'DistilBert': 'distilbert-base-uncased',
        'SqueezeBert': 'squeezebert/squeezebert-uncased',
        'MobileBert': 'google/mobilebert-uncased'
    } 
    
    # error manager
    def _check_and_raise_error(entered, targeted, msg, eq=True):
        if eq:
            if entered == targeted: # raise error if eq(==) condition meets
                err = f'[{args.dataset.upper()}] `{entered}` {msg} is not supported for this dataset!'
                logger.exception(err)
                raise AssertionError(err)
        else:
            if entered != targeted: # raise error if neq(!=) condition meets
                err = f'[{args.dataset.upper()}] `{targeted}` {msg} is only supported for this dataset!'
                logger.exception(err)
                raise AssertionError(err)

    # method to get transformation chain
    def _get_transform(args, train=False):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.resize, args.resize)) if args.resize is not None else torchvision.transforms.Lambda(lambda x: x),
                torchvision.transforms.RandomRotation(args.randrot) if (args.randrot is not None and train) else torchvision.transforms.Lambda(lambda x: x),
                torchvision.transforms.RandomHorizontalFlip(args.randhf) if (args.randhf is not None and train) else torchvision.transforms.Lambda(lambda x: x),
                torchvision.transforms.RandomVerticalFlip(args.randvf) if (args.randvf is not None and train) else torchvision.transforms.Lambda(lambda x: x),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if args.imnorm else torchvision.transforms.Lambda(lambda x: x)
            ]
        )
        return transform

    # method to construct per-client dataset
    def _construct_dataset(raw_train, idx, sample_indices):
        subset = torch.utils.data.Subset(raw_train, sample_indices)
        test_size = int(len(subset) * args.test_fraction)
        training_set, test_set = torch.utils.data.random_split(subset, [len(subset) - test_size, test_size])
        traininig_set = SubsetWrapper(training_set, f'< {str(idx).zfill(8)} > (train)')
        test_set = SubsetWrapper(test_set, f'< {str(idx).zfill(8)} > (test)')
        return {idx: (traininig_set, test_set)}
    
    #################
    # base settings #
    #################
    # required intermediate outputs
    raw_train, raw_test = None, None

    # required outputs
    split_map, client_datasets = None, None
    
    # optional argument for data transforms
    transforms = [None, None]
    
    ####################
    # for text dataset #
    ####################
    tokenizer = None
    if args.use_model_tokenizer or args.use_pt_model:
        assert args.model_name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'

    if args.use_model_tokenizer:
        assert args.model_name.lower() in transformers.models.__dict__.keys(), f'Please check if the model (`{args.model_name}`) is supported by `transformers` module!'
        module = transformers.models.__dict__[f'{args.model_name.lower()}']
        tokenizer = getattr(module, f'{args.model_name}Tokenizer').from_pretrained(TOKENIZER_STRINGS[args.model_name])

    #################
    # fetch dataset #
    #################
    logger.info(f'[LOAD] Fetch dataset!')
    
    if args.dataset in ['FEMNIST', 'Shakespeare', 'Sent140', 'CelebA', 'Reddit']: # 1) for a special dataset - LEAF benchmark...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
         
        # define transform
        if args.dataset in ['FEMNIST', 'CelebA']:
            # check if `resize` is required
            if args.resize is None:
                logger.info(f'[LOAD] Dataset `{args.dataset}` may require `resize` argument; (recommended: `FEMNIST` - 28, `CelebA` - 84)!')
            transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        elif args.dataset == 'Reddit':
            args.rawsmpl = 1.0

        # construct split hashmap, client datasets
        # NOTE: for LEAF benchmark, values of `split_map` hashmap is not indices, but sample counts of tuple (training set, test set)!
        split_map, client_datasets, args = fetch_leaf(
            args=args,
            dataset_name=args.dataset, 
            root=args.data_path, 
            seed=args.seed, 
            raw_data_fraction=args.rawsmpl, 
            test_fraction=args.test_fraction, 
            n_jobs=os.cpu_count() - 1, 
            transforms=transforms
        )

        # no global holdout set for LEAF
        raw_test = None  

    elif args.dataset in torchvision.datasets.__dict__.keys(): # 3) for downloadable datasets in `torchvision.datasets`...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_torchvision_dataset(args=args, dataset_name=args.dataset, root=args.data_path, transforms=transforms)
        
    elif args.dataset in torchtext.datasets.__dict__.keys(): # 4) for downloadable datasets in `torchtext.datasets`...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, args = fetch_torchtext_dataset(args=args, dataset_name=args.dataset, root=args.data_path, seq_len=args.seq_len, tokenizer=tokenizer) 
        
    elif args.dataset == 'TinyImageNet': # 5) for other public datasets...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_tinyimagenet(args=args, root=args.data_path, transforms=transforms)
        
    elif args.dataset == 'CINIC10':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_cinic10(args=args, root=args.data_path, transforms=transforms)
        
    elif 'BeerReviews' in args.dataset:
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        aspect_type = {'A': 'aroma', 'L': 'look'}
        parsed_type = args.dataset[-1]
        if parsed_type in ['A', 'L']:
            aspect = aspect_type[parsed_type]
        else:
            err = '[LOAD] Please check dataset name!'
            logger.exception(err)
            raise Exception(err)
        raw_train, raw_test, args = fetch_beerreviews(args=args, root=args.data_path, aspect=aspect, tokenizer=tokenizer)  
        
    elif args.dataset == 'Heart':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_heart(args=args, root=args.data_path, seed=args.seed, test_fraction=args.test_fraction)
    
    elif args.dataset == 'Adult':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_adult(args=args, root=args.data_path, seed=args.seed, test_fraction=args.test_fraction)
    
    elif args.dataset == 'Cover':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_cover(args=args, root=args.data_path, seed=args.seed, test_fraction=args.test_fraction)  
    
    elif args.dataset == 'GLEAM':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_gleam(args=args, root=args.data_path, seed=args.seed, test_fraction=args.test_fraction, seq_len=args.seq_len)

    else: # x) for a dataset with no support yet or incorrectly entered...
        err = f'[LOAD] Dataset `{args.dataset}` is not supported or seems incorrectly entered... please check!'
        logger.exception(err)
        raise Exception(err)     
    logger.info(f'[LOAD] ...successfully fetched dataset!')
    
    ############
    # finalize #
    ############
    # adjust the number of classes in binary case
    if args.num_classes == 2:
        args.num_classes = 1
        args.criterion = torch.nn.BCEWithLogitsLoss
        
    # check if global holdout set is required or not
    if args.eval_type == 'local':
        raw_test = None
    else:
        if raw_test is None:
            err = f'[LOAD] Dataset `{args.dataset.upper()}` does not support pre-defined evaluation set, please check! (current `eval_type`=`{args.eval_type}`)'
            logger.exception(err)
            raise AssertionError(err)
            
    # get split indices if None
    if split_map is None:
        logger.info(f'[SIMULATE] Simulate dataset split (split scenario: `{args.split_type.upper()}`)!')
        split_map = simulate_split(args, raw_train)
        logger.info(f'[SIMULATE] ...done simulating dataset split (split scenario: `{args.split_type.upper()}`)!')
    
    # construct client datasets if None
    if client_datasets is None:
        logger.info(f'[SIMULATE] Create client datasets!')
        with pool.ThreadPool(processes=os.cpu_count() - 1) as workhorse:
            client_datasets = workhorse.starmap(
                _construct_dataset, 
                [
                    (raw_train, idx, sample_indices) for idx, sample_indices in tqdm(
                        enumerate(split_map.values()),
                        total=len(split_map),
                        leave=False
                    )
                ]
            )
        client_datasets = dict(ChainMap(*client_datasets)) # {client index: (training set, test set)}
        logger.info(f'[SIMULATE] ...created client datasets!')
    return raw_test, client_datasets    
