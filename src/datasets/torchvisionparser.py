import torch
import logging
import torchvision

logger = logging.getLogger(__name__)



# dataset wrapper module
class VisionClassificationDataset(torch.utils.data.Dataset): 
    def __init__(self, dataset, dataset_name, suffix):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.suffix = suffix
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        inputs, targets = self.dataset[index]
        return inputs, targets

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f'[{self.dataset_name}] {self.suffix}'

# helper method to fetch dataset from `torchvision.datasets`
def fetch_torchvision_dataset(args, dataset_name, root, transforms):
    logger.info(f'[LOAD] [{dataset_name.upper()}] Fetching dataset!')
    
    # default arguments
    DEFAULT_ARGS = {'root': root, 'transform': None, 'download': True}
    
    if dataset_name in [
        'MNIST', 'FashionMNIST', 'QMNIST', 'KMNIST', 'EMNIST',\
        'CIFAR10', 'CIFAR100', 'USPS'
    ]:
        # configure arguments for training/test dataset
        train_args = DEFAULT_ARGS.copy()
        train_args['train'] = True
        train_args['transform'] = transforms[0]

        # special case - EMNIST
        if dataset_name == 'EMNIST':
            train_args['split'] = 'byclass'  
            
        # create training dataset instance
        raw_train = torchvision.datasets.__dict__[dataset_name](**train_args)
        raw_train = VisionClassificationDataset(raw_train, dataset_name.upper(), 'CLIENT')

        # for global holdout set
        test_args = DEFAULT_ARGS.copy()
        test_args['transform'] = transforms[1]
        test_args['train'] = False
                
        # special case - EMNIST
        if dataset_name == 'EMNIST':
            test_args['split'] = 'byclass'
                
        # create test dataset instance
        raw_test = torchvision.datasets.__dict__[dataset_name](**test_args)
        raw_test = VisionClassificationDataset(raw_test, dataset_name.upper(), 'SERVER')

        # adjust arguments
        if 'CIFAR' in dataset_name:
            args.in_channels = 3
        else:
            args.in_channels = 1
            
    elif dataset_name in [
        'Country211',\
        'DTD', 'Flowers102', 'Food101', 'FGVCAircraft',\
        'GTSRB', 'RenderedSST2', 'StanfordCars',\
        'STL10', 'SVHN'
    ]:
        # configure arguments for training/test dataset
        train_args = DEFAULT_ARGS.copy() 
        train_args['split'] = 'train'
        train_args['transform'] = transforms[0]

        # create training dataset instance
        raw_train = torchvision.datasets.__dict__[dataset_name](**train_args)
        raw_train = VisionClassificationDataset(raw_train, dataset_name.upper(), 'CLIENT')

        # for global holdout set
        test_args = DEFAULT_ARGS.copy()
        test_args['transform'] = transforms[1]
        test_args['split'] = 'test'
            
        # create test dataset instance
        raw_test = torchvision.datasets.__dict__[dataset_name](**test_args)
        raw_test = VisionClassificationDataset(raw_test, dataset_name.upper(), 'SERVER')

        # for compatibility, create attribute `targets`
        if dataset_name in ['DTD', 'Flowers102', 'Food101', 'FGVCAircraft']:
            setattr(raw_train, 'targets', raw_train._labels)
        elif dataset_name in ['GTSRB', 'RenderedSST2', 'StanfordCars']:
            setattr(raw_train, 'targets', [*list(zip(*raw_train._samples))[-1]])
        elif dataset_name in ['STL10', 'SVHN']:
            setattr(raw_train, 'targets', raw_train.labels)
        
        # adjust arguments
        if 'RenderedSST2' in dataset_name:
            args.in_channels = 1
        else:
            args.in_channels = 3
            
    elif dataset_name in ['Places365', 'INaturalist', 'OxfordIIITPet', 'Omniglot']:
        # configure arguments for training/test dataset
        train_args = DEFAULT_ARGS.copy() 
        train_args['transform'] = transforms[0]

        if dataset_name == 'Places365':
            train_args['split'] = 'train-standard'
        elif dataset_name == 'OxfordIIITPet':
            train_args['split'] = 'trainval'
        elif dataset_name == 'INaturalist':
            train_args['version'] = '2021_train_mini'
        elif dataset_name == 'Omniglot':
            train_args['background'] = True
            
        # create training dataset instance
        raw_train = torchvision.datasets.__dict__[dataset_name](**train_args)
        raw_train = VisionClassificationDataset(raw_train, dataset_name.upper(), 'CLIENT')

        # for global holdout set
        test_args = DEFAULT_ARGS.copy()
        test_args['transform'] = transforms[1]
                
        if dataset_name == 'Places365':
            test_args['split'] = 'val'
        elif dataset_name == 'OxfordIIITPet':
            test_args['split'] = 'test'
        elif dataset_name == 'INaturalist':
            test_args['version'] = '2021_valid'
        elif dataset_name == 'Omniglot':
            test_args['background'] = False
            
        # create test dataset instance
        raw_test = torchvision.datasets.__dict__[dataset_name](**test_args)
        raw_test = VisionClassificationDataset(raw_test, dataset_name.upper(), 'SERVER')
        
        # for compatibility, create attribute `targets` 
        if dataset_name == 'OxfordIIITPet':
            setattr(raw_train, 'targets', raw_train._labels)
        elif dataset_name == 'INaturalist':
            setattr(raw_train, 'targets', [*list(zip(*raw_train.index))[0]])
        elif dataset_name == 'Omniglot':
            setattr(raw_train, 'targets', [*list(zip(*raw_train._flat_character_images))[-1]])
        
        # adjust arguments
        if 'Omniglot' in dataset_name:
            args.in_channels = 1
        else:
            args.in_channels = 3
            
    elif dataset_name in ['Caltech256', 'SEMEION', 'SUN397']:
        # configure arguments for training dataset
        # NOTE: these datasets do NOT provide pre-defined split
        # Thus, use all datasets as a training dataset
        train_args = DEFAULT_ARGS.copy()
        train_args['transform'] = transforms[0]
        
        # create training dataset instance
        raw_train = torchvision.datasets.__dict__[dataset_name](**train_args)
        raw_train = VisionClassificationDataset(raw_train, dataset_name.upper(), 'CLIENT')

        # no holdout set
        raw_test = None
            
        # for compatibility, create attribute `targets` 
        if dataset_name == 'Caltech256':
            setattr(raw_train, 'targets', raw_train.y)
        elif dataset_name == 'SEMEION':
            setattr(raw_train, 'targets', raw_train.labels)
        elif dataset_name == 'SUN397':
            setattr(raw_train, 'targets', raw_train._labels)
            
        # adjust arguments
        if 'SEMEION' in dataset_name:
            args.in_channels = 1
        else:
            args.in_channels = 3
    else:
        err = f'[LOAD] Dataset `{dataset_name}` is not supported!'
        logger.exception(err)
        raise Exception(err)

    args.num_classes = len(torch.unique(torch.as_tensor(raw_train.dataset.targets)))  
    logger.info(f'[LOAD] [{dataset_name.upper()}] ...fetched dataset!')
    return raw_train, raw_test, args
