import os
import torch
import logging
import torchtext
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)



class Adult(torch.utils.data.Dataset):
    def __init__(self, education, inputs, targets, scaler):
        self.identifier = education
        self.inputs, self.targets = inputs, targets
        self.scaler = scaler
    
    @staticmethod
    def inverse_transform(self, inputs):
        return self.scaler.inverse_transform(inputs)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]).float(), torch.tensor(self.targets[index]).long()
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch Adult dataset
def fetch_adult(args, root, seed, test_size):
    URL = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    ]
    MD5 = [
        '5d7c39d7b8804f071cdd1f2a7c460872',
        '35238206dfdf7f1fe215bbb874adecdc'
    ]
    COL_NAME = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',\
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',\
        'house_per_week', 'native_country', 'targets'
    ]
    NUM_COL = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'house_per_week', 'education_num']
    
    def _download(root):
        for idx, (url, md5) in enumerate(zip(URL, MD5)):
            _ = torchtext.utils.download_from_url(
                url=url, 
                root=root, 
                hash_value=md5, 
                hash_type='md5'
            )
            os.rename(os.path.join(root, url.split('/')[-1]), os.path.join(root, f"adult_{'train' if idx == 0 else 'test'}.csv"))
    
    def _munge_and_create_clients(root):
        # load dat
        df = pd.read_csv(os.path.join(root, 'adult_train.csv'), header=None, names=COL_NAME, na_values='?').dropna().reset_index(drop=True)
        df = df.drop(columns=['education'])
    
        # encode categorical data
        for col in df.columns:
            if col not in NUM_COL:
                replace_map = {key: value for value, key in enumerate(sorted(df[col].unique()))}
                df[col] = df[col].replace(replace_map)

        # adjust dtype
        for col in df.columns:
            if col in NUM_COL:
                df[col] = df[col].astype('float')
            else:
                df[col] = df[col].astype('category')
        
        # get one-hot encoded dummy columns for categorical data
        df = pd.concat([pd.get_dummies(df.iloc[:, :-1], columns=[col for col in df.columns if col not in NUM_COL][:-1], drop_first=True, dtype=int), df[['targets']]], axis=1)
        
        # creat clients by education
        clients = {}
        for edu in df['education_num'].unique():
            clients[edu] = df.loc[df['education_num'] == edu]
        return clients
    
    def _process_client_datasets(dataset, seed, test_size):
        # remove identifier column
        edu = int(dataset['education_num'].unique()[0])
        df = dataset.drop(columns=['education_num'])
        inputs, targets = df.iloc[:, :-1].values, df.iloc[:, -1].values
        
        # train-test split with stratified manner
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size, random_state=seed, stratify=targets)
        
        # scaling inputs
        scaler = MinMaxScaler()
        train_inputs[:, :5] = scaler.fit_transform(train_inputs[:, :5])
        test_inputs[:, :5] = scaler.transform(test_inputs[:, :5])
        return (
            Adult(f'[ADULT] CLIENT < Edu{str(edu).zfill(2)} > (train)', train_inputs, train_targets, scaler), 
            Adult(f'[ADULT] CLIENT < Edu{str(edu).zfill(2)} > (test)', test_inputs, test_targets, scaler)
        ) 
        
    logger.info(f'[LOAD] [ADULT] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'adult')):
        _download(root=os.path.join(root, 'adult'))
        logger.info(f'[LOAD] [ADULT] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [ADULT] ...raw data already exists!')
    
    logger.info(f'[LOAD] [ADULT] Munging dataset and create clients!')
    raw_clients = _munge_and_create_clients(os.path.join(root, 'adult'))
    logger.info('[LOAD] [ADULT] ...munged dataset and created clients!!')
    
    logger.info(f'[LOAD] [ADULT] Processing client datsets!')
    client_datasets = []
    for dataset in raw_clients.values():
        client_datasets.append(_process_client_datasets(dataset, seed, test_size))
    logger.info('[LOAD] [ADULT] ...processed client datasets!')
    
    args.in_features = 84
    args.num_classes = 2
    args.K = 16
    return {}, client_datasets, args
