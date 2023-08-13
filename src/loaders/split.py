import logging
import numpy as np

from src import TqdmToLogger

logger = logging.getLogger(__name__)



def simulate_split(args, dataset):
    """Split data indices using labels.
    
    Args:
        args (argparser): arguments
        dataset (dataset): raw dataset instance to be split 
        
    Returns:
        split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices
    """
    # IID split (i.e., statistical homogeneity)
    if args.split_type == 'iid': 
        # shuffle sample indices
        shuffled_indices = np.random.permutation(len(dataset))
        
        # get adjusted indices
        split_indices = np.array_split(shuffled_indices, args.K)
        
        # construct a hashmap
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
    
    # non-IID split by sample unbalancedness
    if args.split_type == 'unbalanced': 
        # shuffle sample indices
        shuffled_indices = np.random.permutation(len(dataset))
        
        # split indices by number of clients
        split_indices = np.array_split(shuffled_indices, args.K)
            
        # randomly remove some proportion (1% ~ 5%) of data
        keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))
            
        # get adjusted indices
        split_indices = [indices[:int(len(indices) * ratio)] for indices, ratio in zip(split_indices, keep_ratio)]
        
        # construct a hashmap
        split_map = {k: split_indices[k] for k in range(args.K)}
        return split_map
    
    # Non-IID split proposed in (McMahan et al., 2016); each client has samples from at least two different classes
    elif args.split_type == 'patho': 
        try:
            assert args.mincls >= 2
        except AssertionError as e:
            logger.exception("[SIMULATE] Each client should have samples from at least 2 distinct classes!")
            raise e
        
        # get indices by class labels
        _, unique_inverse, unique_counts = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_counts[:-1]))
            
        # divide shards
        num_shards_per_class = args.K * args.mincls // args.num_classes
        if num_shards_per_class < 1:
            err = f'[SIMULATE] Increase the number of minimum class (`args.mincls` > {args.mincls}) or the number of participating clients (`args.K` > {args.K})!'
            logger.exception(err)
            raise Exception(err)
        
        # split class indices again into groups, each having the designated number of shards
        split_indices = [np.array_split(np.random.permutation(indices), num_shards_per_class) for indices in class_indices]
        
        # make hashmap to track remaining shards to be assigned per client
        class_shards_counts = dict(zip([i for i in range(args.num_classes)], [len(split_idx) for split_idx in split_indices]))

        # assign divided shards to clients
        assigned_shards = []
        for _ in TqdmToLogger(
            range(args.K), 
            logger=logger,
            desc='[SIMULATE] ...assigning to clients... '
            ):
            # update selection proability according to the count of reamining shards
            # i.e., do NOT sample from class having no remaining shards
            selection_prob = np.where(np.array(list(class_shards_counts.values())) > 0, 1., 0.)
            selection_prob /= sum(selection_prob)
            
            # select classes to be considered
            try:
                selected_classes = np.random.choice(args.num_classes, args.mincls, replace=False, p=selection_prob)
            except: # if shard size is not fit enough, some clients may inevitably have samples from classes less than the number of `mincls`
                selected_classes = np.random.choice(args.num_classes, args.mincls, replace=True, p=selection_prob)
            
            # assign shards in randomly selected classes to current client
            for it, class_idx in enumerate(selected_classes):
                selected_shard_indices = np.random.choice(len(split_indices[class_idx]), 1)[0]
                selected_shards = split_indices[class_idx].pop(selected_shard_indices)
                if it == 0:
                    assigned_shards.append([selected_shards])
                else:
                    assigned_shards[-1].append(selected_shards)
                class_shards_counts[class_idx] -= 1
            else:
                assigned_shards[-1] = np.concatenate(assigned_shards[-1])

        # construct a hashmap
        split_map = {k: assigned_shards[k] for k in range(args.K)}
        return split_map
    
    # Non-IID split proposed in (Hsu et al., 2019); simulation of non-IID split scenario using Dirichlet distribution
    elif args.split_type == 'diri':
        MIN_SAMPLES = int(1 / args.test_size)

        # get indices by class labels
        total_counts = len(dataset.targets)
        _, unique_inverse, unique_counts = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_counts[:-1]))

        # calculate ideal samples counts per client
        ideal_counts = len(dataset.targets) // args.K
        if ideal_counts < 1:
            err = f'[SIMULATE] Decrease the number of participating clients (`args.K` < {args.K})!'
            logger.exception(err)
            raise Exception(err)

        # split dataset
        ## define temporary container
        assigned_indices = []

        ## NOTE: it is possible that not all samples be consumed, as it is intended for satisfying each clients having at least `MIN_SAMPLES` samples per class
        for k in TqdmToLogger(range(args.K), logger=logger, desc='[SIMULATE] ...assigning to clients... '):
            ### for current client of which index is `k`
            curr_indices = []
            satisfied_counts = 0

            ### ...until the number of samples close to ideal counts is filled
            while satisfied_counts < ideal_counts:
                ### define Dirichlet distribution of which prior distribution is an uniform distribution
                diri_prior = np.random.uniform(size=args.num_classes)
                
                ### sample a parameter corresponded to that of categorical distribution
                cat_param = np.random.dirichlet(alpha=args.cncntrtn * diri_prior)

                ### try to sample by amount of `ideal_counts``
                sampled = np.random.choice(args.num_classes, ideal_counts, p=cat_param)

                ### count per-class samples
                unique, counts = np.unique(sampled, return_counts=True)

                ### filter out sampled classes not having as much as `MIN_SAMPLES`
                required_counts = counts * (counts > MIN_SAMPLES)

                ### assign from population indices split by classes 
                for idx, required_class in enumerate(unique):
                    if required_counts[idx] == 0: continue
                    sampled_indices = class_indices[required_class][:required_counts[idx]]
                    curr_indices.append(sampled_indices)
                    class_indices[required_class] = class_indices[required_class][:required_counts[idx]]
                satisfied_counts += sum(required_counts)
            
            ### when enough samples are collected, go to next clients!
            assigned_indices.append(np.concatenate(curr_indices))

        # construct a hashmap
        split_map = {k: assigned_indices[k] for k in range(args.K)}
        return split_map
    # `leaf` - LEAF benchmark (Caldas et al., 2018); `fedvis` - Federated Vision Datasets (Hsu, Qi and Brown, 2020)
    elif args.split_type in ['leaf']: 
        logger.info('[SIMULATE] Use pre-defined split!')
