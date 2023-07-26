import os
import torch
import logging
import torchvision
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class GLEAM(torch.utils.data.Dataset):
    def __init__(self, identifier,inputs, targets):
        self.identifier = identifier
        self.inputs, self.targets = inputs, targets
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]).float(), torch.tensor([self.targets[index]]).long()
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch GLEAM activity classification dataset 
# NOTE: data is split by person
def fetch_gleam(args, root, seed, test_size, seq_len):
    URL = 'http://www.skleinberg.org/data/GLEAM.tar.gz'
    MD5 = '10ad34716546e44c5078d392f25031e1'
    MINMAX = {
        # NOTE: the range is roughly estimated from all collected data 
        ## [(Sensor 1 min, Sensor 1 max), (Sensor 2 min, Senosr 2 max), (Sensor 3 min, Sensor 3 max)]
        0: [(-7, 4.5), (-8.5, 9.5), (-7, 7)], # Gyroscope
        1: [(-18.5, 14.5), (-15, 20), (-14, 17)], # Accelerometer
        2: [(-61, 90), (-111.5, 143), (-115, 49)], # Geomagnetic
        3: [(-1, 1), (-1, 0.95), (-1, 1)], # Rotation vector
        4: [(-18, 14), (-21, 11.5), (-10.5, 18)], # Linear Acceleration
        5: [(-9.9, 9.9), (-9.9, 9.9), (-9.9, 9.9)], # Gravity
        6: [(1, 32675), (0, 0), (0, 0)] # Light
    }

    def _download(root):
        torchvision.datasets.utils.download_and_extract_archive(
            URL, root, filename= URL.split('/')[-1],
            remove_finished=True, md5=MD5
        )
        
        # remove README file
        os.remove(os.path.join(root, 'README_GLEAM.pdf'))

    def _munge_and_split(root, seed, test_size):
        def assign_activity(df, ref_dict):
            for key, value in ref_dict.items():
                if df.name >= value[0] and df.name <= value[1]:
                    return key
        
        # container
        client_datasets = []

        # load raw data
        demo = None
        for idx, (path, dirs, files) in enumerate(os.walk(root)):
            if idx == 0: # demographic information
                demo = pd.read_csv(os.path.join(path, files[0]), usecols=['Id', 'Age', 'Gender', 'Wear glasses?', 'Annotator', 'Chair type'])
                demo = demo.rename(columns={'Wear glasses?': 'Glasses'})
                demo['Annotator'] = demo['Annotator'].apply(lambda x: int(str(x).split(' ')[-1]) - 1)
                demo['Gender'] = demo['Gender'].replace({'M': 0, 'F': 1}).astype(int)
                demo['Glasses'] = demo['Glasses'].replace({'No': 0, 'Yes': 1}).astype(int)
                demo['Chair type'] = demo['Chair type'].replace({'fixed': 0, 'rolling': 1}).astype(int)
            else:
                # get identifier
                identifier = path[-4:]

                # read essential files
                anno, sensor = sorted([file for file in files if not file.startswith('._') and not file.endswith('.jpg') and 'DS_Store' not in file and 'notes' not in file])
                anno, sensor = pd.read_csv(os.path.join(path, anno), usecols=['unix time', 'Activity'], na_values='w').rename(columns={'unix time': 'Time'}), pd.read_csv(os.path.join(path, sensor)).rename(columns={'Unix Time': 'Time'})
                
                # polish sensor name
                sensor['Sensor'] = sensor['Sensor'].str.split().apply(lambda x: x[-1] if len(x) == 2 else x[1]).replace({'Gyroscope': 0, 'Accelerometer': 1, 'Magnetic': 2, 'Rotation': 3, 'Linear': 4, 'Gravity': 5, 'Light': 6})

                # get timestamp
                anno['Time'] = pd.to_datetime(anno['Time'], unit='ms')
                sensor['Time'] = pd.to_datetime(sensor['Time'], unit='ms')

                # convert annotation data to have 'start time', 'end time', corresponding 'activity' type
                anno = pd.DataFrame({'start': anno['Time'].iloc[::2].values, 'end': anno['Time'].iloc[1::2].values, 'Activity': anno['Activity'].iloc[::2].values})
                
                # fill y, z values of luminance sensors with zero
                sensor = sensor.fillna(0)

                # get indices corresponds to start & end time of each activity
                start = sensor['Time'].searchsorted(anno['start'])
                end = sensor['Time'].searchsorted(anno['end']) - 1

                # define activity-index range dictionary
                ref_dict = dict(zip(anno['Activity'].values.tolist(), zip(start, end)))

                # assign activity
                sensor['Activity'] = sensor.apply(lambda x: assign_activity(x, ref_dict), axis=1)

                # mark no activity with 'None'
                sensor['Activity'] = sensor['Activity'].fillna('none')

                # add demographical data
                raw = pd.concat(
                    [
                        sensor.iloc[:, 1:-1], # features (x, y, z sensor)
                        demo[demo['Id'] == int(identifier)].iloc[:, 1:].reset_index(drop=True) 
                    ], axis=1
                ).ffill()
                
                # add target
                raw = pd.concat(
                    [
                        raw,
                        sensor.iloc[:, -1].replace(
                            {
                                'none': -100, # to be ignored by `torch.nn.CrossEntropyLoss` as it is a dominant class 
                                'eat': 0, 
                                'stair': 1, 
                                'walk': 2, 
                                'talk': 3, 
                                'drink': 4, 
                                'play_mobile': 5, 'play_game': 5, 'play game': 5, 'play_cellphone': 5, 
                                'ipad_work': 6, 'computer_work': 6, 'computer work': 6, 'computer': 6, 
                                'study': 7, 'homework': 7, 'home_work': 7
                            }
                        ) # target (activity
                    ], axis=1
                )
                
                # min-max normalize based on sensor range
                for c in range(len(MINMAX)):
                    (xmin, xmax), (ymin, ymax), (zmin, zmax) = MINMAX[c]
                    raw.loc[raw['Sensor'] == c, 'Value1'] = (raw.loc[raw['Sensor'] == c, 'Value1'] - xmin) / (xmax - xmin)
                    raw.loc[raw['Sensor'] == c, 'Value2'] = (raw.loc[raw['Sensor'] == c, 'Value2'] - ymin) / (ymax - ymin)
                    raw.loc[raw['Sensor'] == c, 'Value3'] = (raw.loc[raw['Sensor'] == c, 'Value3'] - zmin) / (zmax - zmin)
                else:
                    raw = raw.fillna(0)

                # scale 'Age' variable (18 ~ 21)
                raw['Age'] -= 18
                raw['Age'] /= (21 - 18)

                # one-hot encode sensor type
                raw = pd.concat([pd.get_dummies(raw['Sensor'], drop_first=True, prefix='Sensor', prefix_sep=''), raw.iloc[:, 1:]], axis=1)

                # split into train &  test set
                train, test = raw.iloc[:-int(len(raw) * test_size)], raw.iloc[-int(len(raw) * test_size):]
                
                # split into inputs and targets
                train_inputs, train_targets = train.iloc[:, :-1].values, train.iloc[:, -1].values
                test_inputs, test_targets = test.iloc[:, :-1].values, test.iloc[:, -1].values
                
                # slice by `seq_len`
                train_inputs_chunks, train_targets_chunks = np.split(train_inputs, np.arange(seq_len, len(train_inputs), seq_len)), np.split(train_targets, np.arange(seq_len, len(train_targets), seq_len))
                test_inputs_chunks, test_targets_chunks = np.split(test_inputs, np.arange(seq_len, len(test_inputs), seq_len)), np.split(test_targets, np.arange(seq_len, len(test_targets), seq_len))

                # pad
                train_inputs_chunks[-1] = np.pad(train_inputs_chunks[-1], ((0, seq_len - len(train_inputs_chunks[-1])), (0, 0)), mode='constant', constant_values=0)
                train_targets_chunks[-1] = np.pad(train_targets_chunks[-1], (seq_len - len(train_targets_chunks[-1]), 0), mode='constant', constant_values=-100)

                test_inputs_chunks[-1] = np.pad(test_inputs_chunks[-1], ((0, seq_len - len(test_inputs_chunks[-1])), (0, 0)), mode='constant', constant_values=0)
                test_targets_chunks[-1] = np.pad(test_targets_chunks[-1], (seq_len - len(test_targets_chunks[-1]), 0), mode='constant', constant_values=-100)

                # re-assign padded inputs: i.e., (num_chunks, seq_len, in_features)
                train_inputs, train_targets = np.array(train_inputs_chunks).astype(float), np.array(train_targets_chunks).astype(int)
                test_inputs, test_targets = np.array(test_inputs_chunks).astype(float), np.array(test_targets_chunks).astype(int)

                # assign as a client dataset
                client_datasets.append(
                    (
                        GLEAM(f'[GLEAM] CLIENT < {identifier} > (train)', train_inputs, train_targets), 
                        GLEAM(f'[GLEAM] CLIENT < {identifier} > (test)', test_inputs, test_targets)
                    )
                )
        return client_datasets

    logger.info(f'[LOAD] [GLEAM] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'gleam')):
        _download(root=os.path.join(root, 'gleam'))
        logger.info(f'[LOAD] [GLEAM] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [GLEAM] ...raw data already exists!')
    
    logger.info(f'[LOAD] [GLEAM] Munging and splitting dataset!')
    client_datasets = _munge_and_split(os.path.join(root, 'gleam'), seed, test_size)
    logger.info('[LOAD] [GLEAM] ...munged and splitted dataset!')
    
    args.in_features = 14
    args.num_classes = 8
    args.K = 38
    return {}, client_datasets, args
