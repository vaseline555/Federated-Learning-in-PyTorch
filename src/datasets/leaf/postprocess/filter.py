import os
import json



def filter_clients(dataset_name, root, min_samples_per_clients):
    # set path
    data_dir = os.path.join(root, dataset_name)
    subdir = os.path.join(data_dir, 'sampled_data')
    
    # collect sampled files
    files = []
    if os.path.exists(subdir):
        files = os.listdir(subdir)
    if len(files) == 0:
        subdir = os.path.join(data_dir, 'all_data')
        files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]

    # calculate remaining clients data
    for f in files:
        users = []
        hierarchies = []
        num_samples = []
        user_data = {}

        file_dir = os.path.join(subdir, f)
        with open(file_dir, 'r') as file:
            data = json.load(file)

        num_users = len(data['users'])
        for i in range(num_users):
            curr_user = data['users'][i]
            curr_hierarchy = None
            if 'hierarchies' in data:
                curr_hierarchy = data['hierarchies'][i]
            curr_num_samples = data['num_samples'][i]
            if curr_num_samples >= min_samples_per_clients:
                user_data[curr_user] = data['user_data'][curr_user]
                users.append(curr_user)
                if curr_hierarchy is not None:
                    hierarchies.append(curr_hierarchy)
                num_samples.append(data['num_samples'][i])
        
        # create json file
        all_data = {}
        all_data['users'] = users
        if len(hierarchies) == len(users):
            all_data['hierarchies'] = hierarchies
        all_data['num_samples'] = num_samples
        all_data['user_data'] = user_data

        # save file
        with open(os.path.join(data_dir, 'rem_clients_data', f'{f[:-5]}_keep_{str(min_samples_per_clients).zfill(4)}.json'), 'w') as outfile:
            json.dump(all_data, outfile)
