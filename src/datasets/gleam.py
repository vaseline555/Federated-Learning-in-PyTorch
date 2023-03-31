import os
import torch
import logging
import torchvision
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class GLEAM(torch.utils.data.Dataset):
    def __init__(self, identifier,inputs, targets, scaler):
        self.identifier = identifier
        self.inputs, self.targets = inputs, targets
        self.scaler = scaler
    
    @staticmethod
    def inverse_transform(self, inputs):
        return self.scaler.inverse_transform(inputs)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]), torch.tensor([self.targets[index]]).long()
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch GLEAM activity classification dataset 
# NOTE: data is split by person
def fetch_gleam(args, root, seed, test_fraction):
    URL = 'http://www.skleinberg.org/data/GLEAM.tar.gz'
    MD5 = '10ad34716546e44c5078d392f25031e1'
    COL_NAME = []

    def _download(root):
        torchvision.datasets.utils.download_and_extract_archive(
            URL, root, filename= URL.split('/')[-1],
            remove_finished=True, md5=MD5
        )
        os.rename(os.path.join(root, 'covtype.data'), os.path.join(root, 'covtype.csv'))
    
    def _munge_and_split(root, seed, test_fraction):
        # load data
        df = pd.read_csv(os.path.join(root, 'covtype.csv'), header=None)
        
        # reverse one-hot encoded columns
        wilderness_area = pd.Series(df.iloc[:, 10:14].values.argmax(1))

        # concatenate into one dataframe
        df_raw = pd.concat([df.iloc[:, :10], wilderness_area, df.iloc[:, 14:-1], df.iloc[:, -1]], axis=1)

        # rename column
        df_raw.columns = COL_NAME

        # split by wilderness area
        client_datasets = dict()
        for idx, name in enumerate(AREA):
            # get dataframe
            df_temp = df_raw[df_raw['wilderness_area'] == idx].reset_index(drop=True)

            # get inputs and targets
            inputs, targets = df_temp.iloc[:, :-1].values.astype('float'), df_temp.iloc[:, -1].values.astype('float')
        
            # train-test split with stratified manner
            train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_fraction, random_state=seed, stratify=targets)

            # scaling inputs
            scaler = StandardScaler()
            train_inputs[:, :-40] = scaler.fit_transform(train_inputs[:, :-40]) # exclude last 40 columns (`soil type` - one-hot encoded categorical variable)
            test_inputs[:, :-40] = scaler.transform(test_inputs[:, :-40])

            # assign as a client dataset
            client_datasets[idx] = (
                GLEAM(f'[GLEAM] CLIENT < {name} > (train)', train_inputs, train_targets, scaler), 
                GLEAM(f'[GLEAM] CLIENT < {name} > (test)', test_inputs, test_targets, scaler)
            )
        return client_datasets

    logger.info(f'[LOAD] [GLEAM] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'gleam')):
        _download(root=os.path.join(root, 'gleam'))
        logger.info(f'[LOAD] [GLEAM] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [GLEAM] ...raw data already exists!')
    
    logger.info(f'[LOAD] [GLEAM] Munging and splitting dataset!')
    client_datasets = _munge_and_split(os.path.join(root, 'gleam'), seed, test_fraction)
    logger.info('[LOAD] [GLEAM] ...munged and splitted dataset!')
    
    args.in_features = 51
    args.num_classes = 7
    args.K = 38
    return {}, client_datasets, args
