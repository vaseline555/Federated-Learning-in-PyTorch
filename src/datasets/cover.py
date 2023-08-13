import os
import torch
import logging
import torchvision
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class Cover(torch.utils.data.Dataset):
    def __init__(self, groupby, inputs, targets, scaler):
        self.identifier = groupby
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

# helper method to fetch Cover type classification dataset 
# NOTE: data is grouped and split by `wilderness_area`
def fetch_cover(args, root, seed, test_size):
    URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    MD5 = '99670d8d942f09d459c7d4486fca8af5'
    COL_NAME = [
        'elevation', 'aspect', 'slope',\
        'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways',\
        'hillshade_9am', 'hiillshade_noon', 'hillshade_3pm',\
        'horizontal_distance_to_fire_points',\
        'wilderness_area',\
        'soil_type_0', 'soil_type_1', 'soil_type_2', 'soil_type_3', 'soil_type_4',\
        'soil_type_5', 'soil_type_6', 'soil_type_7', 'soil_type_8', 'soil_type_9',\
        'soil_type_10', 'soil_type_11', 'soil_type_12', 'soil_type_13', 'soil_type_14',\
        'soil_type_15', 'soil_type_16', 'soil_type_17', 'soil_type_18', 'soil_type_19',\
        'soil_type_20', 'soil_type_21', 'soil_type_22', 'soil_type_23', 'soil_type_24',\
        'soil_type_25', 'soil_type_26', 'soil_type_27', 'soil_type_28', 'soil_type_29',\
        'soil_type_30', 'soil_type_31', 'soil_type_32', 'soil_type_33', 'soil_type_34',\
        'soil_type_35', 'soil_type_36', 'soil_type_37', 'soil_type_38', 'soil_type_39',\
        'cover_type'
    ]
    AREA = ['Rawah', 'Neota', 'Comanche Peak', 'Cache la Poudre']

    def _download(root):
        torchvision.datasets.utils.download_and_extract_archive(
            URL, root, filename= URL.split('/')[-1],
            remove_finished=True, md5=MD5
        )
        os.rename(os.path.join(root, 'covtype.data'), os.path.join(root, 'covtype.csv'))
    
    def _munge_and_split(root, seed, test_size):
        # load data
        df = pd.read_csv(os.path.join(root, 'covtype.csv'), header=None)
        
        # reverse one-hot encoded columns
        wilderness_area = pd.Series(df.iloc[:, 10:14].values.argmax(1))

        # concatenate into one dataframe
        df_raw = pd.concat([df.iloc[:, :10], wilderness_area, df.iloc[:, 14:-1], df.iloc[:, -1].sub(1)], axis=1)

        # rename column
        df_raw.columns = COL_NAME

        # split by wilderness area
        client_datasets = []
        for idx, name in enumerate(AREA):
            # get dataframe
            df_temp = df_raw[df_raw['wilderness_area'] == idx].reset_index(drop=True)

            # get inputs and targets
            inputs, targets = df_temp.iloc[:, :-1].values.astype('float'), df_temp.iloc[:, -1].values.astype('float')
        
            # train-test split with stratified manner
            train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size, random_state=seed, stratify=targets)

            # scaling inputs
            scaler = StandardScaler()
            train_inputs[:, :-40] = scaler.fit_transform(train_inputs[:, :-40]) # exclude last 40 columns (`soil type` - one-hot encoded categorical variable)
            test_inputs[:, :-40] = scaler.transform(test_inputs[:, :-40])

            # assign as a client dataset
            client_datasets.append(
                (
                    Cover(f'[COVER] CLIENT < {name} > (train)', train_inputs, train_targets, scaler), 
                    Cover(f'[COVER] CLIENT < {name} > (test)', test_inputs, test_targets, scaler)
                )
            )
        return client_datasets

    logger.info(f'[LOAD] [COVER] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'covertype')):
        _download(root=os.path.join(root, 'covertype'))
        logger.info(f'[LOAD] [COVER] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [COVER] ...raw data already exists!')
    
    logger.info(f'[LOAD] [COVER] Munging and splitting dataset!')
    client_datasets = _munge_and_split(os.path.join(root, 'covertype'), seed, test_size)
    logger.info('[LOAD] [COVER] ...munged and splitted dataset!')
    
    args.in_features = 51
    args.num_classes = 7
    args.K = 4
    return {}, client_datasets, args
