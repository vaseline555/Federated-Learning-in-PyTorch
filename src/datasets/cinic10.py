import os
import torch
import logging
import torchvision

logger = logging.getLogger(__name__)



# dataset wrapper module
class CINIC10(torchvision.datasets.ImageFolder):
    base_folder = 'cinic-10-batches-py'
    zip_md5 ='6ee4d0c996905fe93221de577967a372'
    splits = ('train', 'val', 'test')
    filename = 'CINIC-10.tar.gz'
    url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'

    def __init__(self, root, split='train', download=True, transform=None, **kwargs):
        self.data_root = os.path.expanduser(root)
        self.split = torchvision.datasets.utils.verify_str_arg(split, 'split', self.splits)
        if download:
            self.download()
        if not self._check_exists():
            err = 'Dataset not found or corrupted. You can use download=True to download it'
            logger.exception(err)
            raise RuntimeError(err)
        super().__init__(root=self.split_folder, transform=transform, **kwargs)
    
    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def download(self):
        if self._check_exists(): 
            return
        torchvision.datasets.utils.download_and_extract_archive(
            self.url, self.dataset_folder, filename=self.filename,
            remove_finished=True, md5=self.zip_md5
        )
    
    def __repr__(self):
        rep_str = {'train': 'CLIENT', 'test': 'SERVER'}
        return f'[CINIC10] {rep_str[self.split]}'
        
# helper method to fetch CINIC-10 dataset
def fetch_cinic10(args, root, transforms):
    logger.info('[LOAD] [CINIC10] Fetching dataset!')
    
    # default arguments
    DEFAULT_ARGS = {'root': root, 'transform': None, 'download': True}
    
    # configure arguments for training/test dataset
    train_args = DEFAULT_ARGS.copy()
    train_args['split'] = 'train'
    train_args['transform'] = transforms[0]

    # create training dataset instance
    raw_train = CINIC10(**train_args)
        
    # for global holdout set
    test_args = DEFAULT_ARGS.copy()
    test_args['transform'] = transforms[1]
    test_args['split'] = 'test'
            
    # create test dataset instance
    raw_test = CINIC10(**test_args)
    
    logger.info('[LOAD] [CINIC10] ...fetched dataset!')
    
    # adjust arguments
    args.in_channels = 3
    args.num_classes = len(torch.unique(torch.as_tensor(raw_train.targets)))  
    return raw_train, raw_test, args
