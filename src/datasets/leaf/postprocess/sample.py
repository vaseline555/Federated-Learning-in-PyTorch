import os
import json
import random

from collections import OrderedDict



def sample_clients(dataset_name, root, seed, used_raw_data_fraction):
    # set path
    data_dir = os.path.join(root, dataset_name)
    subdir = os.path.join(data_dir, 'all_data')
    files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]
    
    # set seed
    rng = random.Random(seed)
    
    # split data in non-IID manner
    for f in files:
        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as file:
            data = json.load(file, object_pairs_hook=OrderedDict)

        # get meta data
        num_users = len(data['users'])
        tot_num_samples = sum(data['num_samples'])
        num_new_samples = int(used_raw_data_fraction * tot_num_samples)
        hierarchies = None
    
        # non-IID split
        ctot_num_samples = 0
        users = data['users']
        users_and_hiers = None
        if 'hierarchies' in data:
            users_and_hiers = list(zip(users, data['hierarchies']))
            rng.shuffle(users_and_hiers)
        else:
            rng.shuffle(users)
        user_i = 0
        num_samples = []
        user_data = {}

        if 'hierarchies' in data:
            hierarchies = []
        
        while(ctot_num_samples < num_new_samples):
            hierarchy = None
            if users_and_hiers is not None:
                user, hier = users_and_hiers[user_i]
            else:
                user = users[user_i]
            cdata = data['user_data'][user]
            cnum_samples = len(data['user_data'][user]['y'])

            if (ctot_num_samples + cnum_samples) > num_new_samples:
                cnum_samples = num_new_samples - ctot_num_samples
                indices = [i for i in range(cnum_samples)]
                new_indices = rng.sample(indices, cnum_samples)
                x, y = [], []
                for i in new_indices:
                    x.append(data['user_data'][user]['x'][i])
                    y.append(data['user_data'][user]['y'][i])
                cdata = {'x': x, 'y': y}
            
            if 'hierarchies' in data:
                hierarchies.append(hier)

            num_samples.append(cnum_samples)
            user_data[user] = cdata
            ctot_num_samples += cnum_samples
            user_i += 1
            
        if 'hierarchies' in data:
            users = [u for u, h in users_and_hiers][:user_i]
        else:
            users = users[:user_i]

        # create json file
        all_data = {}
        all_data['users'] = users
        if hierarchies is not None:
            all_data['hierarchies'] = hierarchies
        all_data['num_samples'] = num_samples
        all_data['user_data'] = user_data

        # save file
        with open(os.path.join(data_dir, 'sampled_data', f'{f[:-5]}_niid_0{str(used_raw_data_fraction)[2:]}.json'), 'w') as out_file:
            json.dump(all_data, out_file)
