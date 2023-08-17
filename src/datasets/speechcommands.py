import os
import torch
import logging
import torchaudio

logger = logging.getLogger(__name__)


LABELS = sorted([
    'on', 'learn', 'tree', 'down', 'forward', 
    'backward', 'happy', 'off', 'nine', 'eight', 
    'left', 'four', 'one', 'visual', 'sheila', 
    'no', 'six', 'dog', 'up', 'five', 
    'marvin', 'cat', 'yes', 'zero', 'house', 
    'bird', 'go', 'seven', 'stop', 'wow', 
    'three', 'follow', 'right', 'bed', 'two'
])

# dataset wrapper module
class AudioClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dataset_name, suffix):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.suffix = suffix
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        def _label_to_index(word):
            # Return the position of the word in labels
            return torch.tensor(LABELS.index(word))

        def _pad_sequence(batch, max_len=16000):
            # Make all tensor in a batch the same length by padding with zeros
            batch = [torch.nn.functional.pad(item.t(), (0, 0, 0, max_len - len(item.t())), value=0.) for item in batch]
            batch = torch.cat(batch)
            return batch.t()

        # get raw batch by index
        batch = self.dataset[index]

        # gather in lists, and encode labels as indices
        inputs, targets = [], []
        for waveform, _, label, *_ in (batch,):
            inputs += [waveform]
            targets += [_label_to_index(label)]

        # group the list of tensors into a batched tensor
        inputs = _pad_sequence(inputs)
        targets = torch.stack(targets).squeeze()
        return inputs, targets

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f'[{self.dataset_name}] {self.suffix}'

class SpeechCommands(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, root, split, download):
        self.data_root = os.path.expanduser(root)
        super(SpeechCommands, self).__init__(root=self.data_root, subset=split, download=download)

    def __repr__(self):
        rep_str = {'train': 'CLIENT', 'test': 'SERVER'}
        return f'[SpeechCommands] {rep_str[self.split]}'

# helper method to fetch CINIC-10 dataset
def fetch_speechcommands(args, root):
    logger.info('[LOAD] [SpeechCommands] Fetching dataset!')
    
    # default arguments
    DEFAULT_ARGS = {'root': root, 'download': True}
    
    # configure arguments for training/test dataset
    train_args = DEFAULT_ARGS.copy()
    train_args['split'] = 'training'

    # create training dataset instance
    raw_train = SpeechCommands(**train_args)
    train_targets = torch.tensor([LABELS.index(filename.split('/')[3]) for filename in raw_train._walker]).long()
    setattr(raw_train, 'targets', train_targets)
    raw_train = AudioClassificationDataset(raw_train, 'SpeechCommands'.upper(), 'CLIENT')

    # for global holdout set
    test_args = DEFAULT_ARGS.copy()
    test_args['split'] = 'testing'
            
    # create test dataset instance
    raw_test = SpeechCommands(**test_args)
    test_targets = torch.tensor([LABELS.index(filename.split('/')[3]) for filename in raw_test._walker]).long()
    setattr(raw_test, 'targets', test_targets)
    raw_test = AudioClassificationDataset(raw_test, 'SpeechCommands'.upper(), 'SERVER')
    logger.info('[LOAD] [SpeechCommands] ...fetched dataset!')
    
    # adjust arguments
    args.in_channels = 1
    args.num_classes = len(torch.unique(torch.as_tensor(raw_train.dataset.targets)))  
    return raw_train, raw_test, args


