import os
import json
import logging

from src.datasets.leaf.postprocess.sample import sample_clients
from src.datasets.leaf.postprocess.filter import filter_clients
from src.datasets.leaf.postprocess.split import split_datasets

logger = logging.getLogger(__name__)



def postprocess_leaf(dataset_name, root, seed, raw_data_fraction, min_samples_per_clients, test_fraction):
    # check if raw data is prepared 
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Check pre-processing data...!')
    if not os.path.exists(f'{root}/{dataset_name}/all_data'):
        err = f'[LOAD] [LEAF - {dataset_name.upper()}] Please check if the raw data is correctly prepared in `{root}`!'
        raise AssertionError(err)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...data pre-proceesing has been completed!')
    
    # create client datasets
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Sample clients from raw data...!')
    if not os.path.exists(f'{root}/{dataset_name}/sampled_data'):
        os.makedirs(f'{root}/{dataset_name}/sampled_data')
        sample_clients(dataset_name, root, seed, raw_data_fraction)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...done sampling clients from raw data!')
    
    # remove clients with less than given `min_samples_per_clients`
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Filter out remaining clients...!')
    if not os.path.exists(f'{root}/{dataset_name}/rem_clients_data') and (raw_data_fraction < 1.):
        os.makedirs(f'{root}/{dataset_name}/rem_clients_data')
        filter_clients(dataset_name, root, min_samples_per_clients)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...done filtering remaining clients!')
    
    # create train-test split
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] Split into training & test sets...!')
    if (not os.path.exists(f'{root}/{dataset_name}/train')) or (not os.path.exists(f'{root}/{dataset_name}/test')):
        if not os.path.exists(f'{root}/{dataset_name}/train'):
            os.makedirs(f'{root}/{dataset_name}/train')
        if not os.path.exists(f'{root}/{dataset_name}/test'):
            os.makedirs(f'{root}/{dataset_name}/test')    
        split_datasets(dataset_name, root, seed, test_fraction)
    logger.info(f'[LOAD] [LEAF - {dataset_name.upper()}] ...done splitting into training & test sets!')
    
    # get number of clients
    all_data = [file for file in os.listdir(os.path.join(root, dataset_name, 'all_data')) if file.endswith('.json')][0]
    num_clients = len(json.load(open(f'{root}/{dataset_name}/all_data/{all_data}', 'r'))['users'])
    return num_clients
