import os
import sys
import json
import random
import logging

from collections import OrderedDict



def split_datasets(dataset_name, root, seed, test_fraction):
    # set path
    data_dir = os.path.join(root, dataset_name)
    subdir = os.path.join(data_dir, 'rem_clients_data')
    
    # collect sampled files
    files = []
    if os.path.exists(subdir):
        files = os.listdir(subdir)
    if len(files) == 0:
        subdir = os.path.join(data_dir, 'sampled_data')
        if os.path.exists(subdir):
            files = os.listdir(subdir)
    if len(files) == 0:
        subdir = os.path.join(data_dir, 'all_data')
        files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]
    
    # set seed
    rng = random.Random(seed)

    # check if data contains information on hierarchies
    file_dir = os.path.join(subdir, files[0])
    with open(file_dir, 'r') as file:
        data = json.load(file)
    
    # split training/test data inside clients
    for f in files:
        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as file:
            data = json.load(file, object_pairs_hook=OrderedDict)

        num_samples_train, num_samples_test = [], []
        user_data_train, user_data_test = {}, {}
        user_indices = [] # indices of users in `data['users']` that are not deleted

        for i, u in enumerate(data['users']):
            curr_num_samples = len(data['user_data'][u]['y'])
            if curr_num_samples >= 2:
                # ensures number of train and test samples both >= 1
                num_train_samples = max(1, int((1. - test_fraction) * curr_num_samples))
                if curr_num_samples == 2:
                    num_train_samples = 1
                num_test_samples = curr_num_samples - num_train_samples
                
                indices = [j for j in range(curr_num_samples)]
                if dataset_name == 'shakespeare':
                    train_indices = [i for i in range(num_train_samples)]
                    test_indices = [i for i in range(num_train_samples + 80 - 1, curr_num_samples)]
                else:
                    train_indices = rng.sample(indices, num_train_samples)
                    test_indices = [i for i in range(curr_num_samples) if i not in train_indices]

                if len(train_indices) >= 1 and len(test_indices) >= 1:
                    user_indices.append(i)
                    num_samples_train.append(num_train_samples)
                    num_samples_test.append(num_test_samples)
                    user_data_train[u] = {'x': [], 'y': []}
                    user_data_test[u] = {'x': [], 'y': []}

                    train_blist = [False for _ in range(curr_num_samples)]
                    test_blist = [False for _ in range(curr_num_samples)]

                    for j in train_indices: train_blist[j] = True
                    for j in test_indices:test_blist[j] = True

                    for j in range(curr_num_samples):
                        if train_blist[j]:
                            user_data_train[u]['x'].append(data['user_data'][u]['x'][j])
                            user_data_train[u]['y'].append(data['user_data'][u]['y'][j])
                        elif test_blist[j]:
                            user_data_test[u]['x'].append(data['user_data'][u]['x'][j])
                            user_data_test[u]['y'].append(data['user_data'][u]['y'][j])
        users = [data['users'][i] for i in user_indices]
        
        # create json file of training set
        all_data_train = {}
        all_data_train['users'] = users
        all_data_train['num_samples'] = num_samples_train
        all_data_train['user_data'] = user_data_train
    
        # save file of training set
        with open(os.path.join(data_dir, 'train', f'{f[:-5]}_train_0{str(1. - test_fraction)[2:]}.json'), 'w') as outfile:
            json.dump(all_data_train, outfile)
        
        # create json file of test set
        all_data_test = {}
        all_data_test['users'] = users
        all_data_test['num_samples'] = num_samples_test
        all_data_test['user_data'] = user_data_test
        
        # save file of test set
        with open(os.path.join(data_dir, 'test', f'{f[:-5]}_test_0{str(test_fraction)[2:]}.json'), 'w') as outfile:
            json.dump(all_data_test, outfile)
