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
        _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
            
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
        def sample_with_mask(mask, ideal_samples_counts, concentration, num_classes, need_adjustment=False):
            num_remaining_classes = int(mask.sum())
            
            # sample class selection probabilities based on Dirichlet distribution with concentration parameter (`diri_alpha`)
            selection_prob_raw = np.random.dirichlet(alpha=np.ones(num_remaining_classes) * concentration, size=1).squeeze()
            selection_prob = mask.copy()
            selection_prob[selection_prob == 1.] = selection_prob_raw
            selection_prob /= selection_prob.sum()

            # calculate per-class sample counts based on selection probabilities
            if need_adjustment: # if remaining samples are not enough, force adjusting sample sizes...
                selected_counts = (selection_prob * ideal_samples_counts * np.random.uniform(low=0.0, high=1.0, size=len(selection_prob))).astype(int)
            else:
                selected_counts = (selection_prob * ideal_samples_counts).astype(int)
            return selected_counts
            
        # get indices by class labels
        _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
        class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
        
        # make hashmap to track remaining samples per class
        class_samples_counts = dict(zip([i for i in range(args.num_classes)], [len(class_idx) for class_idx in class_indices]))
        
        # calculate ideal samples counts per client
        ideal_samples_counts = len(dataset.targets) // args.K
        if ideal_samples_counts < 1:
            err = f'[SIMULATE] Decrease the number of participating clients (`args.K` < {args.K})!'
            logger.exception(err)
            raise Exception(err)

        # assign divided shards to clients
        assigned_indices = []
        for k in TqdmToLogger(
            range(args.K), 
            logger=logger,
            desc='[SIMULATE] ...assigning to clients... '
            ):
            # update mask according to the count of reamining samples per class
            # i.e., do NOT sample from class having no remaining samples
            remaining_mask = np.where(np.array(list(class_samples_counts.values())) > 0, 1., 0.)
            selected_counts = sample_with_mask(remaining_mask, ideal_samples_counts, args.cncntrtn, args.num_classes)

            # check if enough samples exist per selected class
            expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)
            valid_mask = np.where(expected_counts < 0, 1., 0.)
            
            # if not, resample until enough samples are secured
            while sum(valid_mask) > 0:
                # resample from other classes instead of currently selected ones
                adjusted_mask = (remaining_mask.astype(bool) & (~valid_mask.astype(bool))).astype(float)
                
                # calculate again if enoush samples exist or not
                selected_counts = sample_with_mask(adjusted_mask, ideal_samples_counts, args.cncntrtn, args.num_classes, need_adjustment=True)    
                expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)

                # update mask for checking a termniation condition
                valid_mask = np.where(expected_counts < 0, 1., 0.)
                
            # assign shards in randomly selected classes to current client
            indices = []
            for it, counts in enumerate(selected_counts):
                # get indices from the selected class
                selected_indices = class_indices[it][:counts]
                indices.extend(selected_indices)
                
                # update indices and statistics
                class_indices[it] = class_indices[it][counts:]
                class_samples_counts[it] -= counts
            else:
                assigned_indices.append(indices)

        # construct a hashmap
        split_map = {k: assigned_indices[k] for k in range(args.K)}
        return split_map
    # `leaf` - LEAF benchmark (Caldas et al., 2018); `fedvis` - Federated Vision Datasets (Hsu, Qi and Brown, 2020)
    elif args.split_type in ['leaf', 'fedvis']: 
        logger.info('[SIMULATE] Use pre-defined split!')
