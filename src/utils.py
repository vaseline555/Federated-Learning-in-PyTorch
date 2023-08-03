import os
import sys
import json
import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from multiprocessing import Process

logger = logging.getLogger(__name__)



#########################
# Argparser Restriction #
#########################
class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end

########
# Seed #
########
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f'[SEED] ...seed is set: {seed}!')
    
###############
# TensorBaord #
###############
class TensorBoardRunner:
    def __init__(self, path, host, port):
        logger.info('[TENSORBOARD] Start TensorBoard process!')
        self.server = TensorboardServer(path, host, port)
        self.server.start()
        self.daemon = True
         
    def finalize(self):
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        self.server.pkill()
        logger.info('[TENSORBOARD] ...finished TensorBoard process!')
        
    def interrupt(self):
        self.server.pkill()
        if self.server.is_alive():    
            self.server.terminate()
            self.server.join()
        logger.info('[TENSORBOARD] ...interrupted; killed all TensorBoard processes!')

class TensorboardServer(Process):
    def __init__(self, path, host, port):
        super().__init__()
        self.os_name = os.name
        self.path = str(path)
        self.host = host
        self.port = port
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --reuse_port=true --port {self.port} 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.path}" --host {self.host} --reuse_port=true --port {self.port} >/dev/null 2>&1')
        else:
            err = f'Current OS ({self.os_name}) is not supported!'
            logger.exception(err)
            raise Exception(err)
    
    def pkill(self):
        if self.os_name == 'nt':
            os.system(f'taskkill /IM "tensorboard.exe" /F')
        elif self.os_name == 'posix':
            os.system('pgrep -f tensorboard | xargs kill -9')

###############
# tqdm add-on #
###############
class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None, 
    mininterval=0.1, 
    bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]', 
    desc=None, 
    **kwargs
    ):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval, bar_format=bar_format, desc=desc, **kwargs)

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight'):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'truncnorm':
                torch.nn.init.trunc_normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

#####################
# Arguments checker #
#####################
def check_args(args):
    # check device
    if 'cuda' in args.device:
        assert torch.cuda.is_available(), 'Please check if your GPU is available now!' 

    # check optimizer
    if args.optimizer not in torch.optim.__dict__.keys():
        err = f'`{args.optimizer}` is not a submodule of `torch.optim`... please check!'
        logger.exception(err)
        raise AssertionError(err)
    
    # check criterion
    if args.criterion not in torch.nn.__dict__.keys():
        err = f'`{args.criterion}` is not a submodule of `torch.nn`... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check algorithm
    if args.algorithm == 'fedsgd':
        args.E = 1
        args.B = 0
    elif args.algorithm == 'fedavgm':
        if args.beta <= 0:
            err = f'server momentum factor (i.e., `beta`) should be positive... please check!'
            logger.exception(err)
            raise AssertionError(err)
    if args.beta > 0:
        args.algorithm = 'fedavgm'
    
    # check model
    if args.model_name == 'Sent140LSTM':
        with open(os.path.join(args.data_path, 'sent140', 'vocab', 'glove.6B.300d.json'), 'r') as file:
            emb_weights = torch.tensor(json.load(file))
        args.glove_emb = emb_weights
    else:
        args.glove_emb = False

    # check lr step
    if args.lr_decay_step > args.R:
        err = f'step size for learning rate decay (`{args.lr_decay_step}`) should be smaller than total round (`{args.R}`)... please check!'
        logger.exception(err)
        raise AssertionError(err)

    # check train only mode
    if args.test_size == 0:
        args._train_only = True
    else:
        args._train_only = False

    # check compatibility of evaluation metrics
    if hasattr(args, 'num_classes'):
        if args.num_classes > 2:
            if ('auprc' or 'youdenj') in args.eval_metrics:
                err = f'some metrics (`auprc`, `youdenj`) are not compatible with multi-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)
        else:
            if 'acc5' in args.eval_metrics:
                err = f'Top5 accruacy (`acc5`) is not compatible with binary-class setting... please check!'
                logger.exception(err)
                raise AssertionError(err)

        if ('mse' or 'mae' or 'mape' or 'rmse' or 'r2' or 'd2') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a classification task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)
    else:
        if ('acc1' or 'acc5' or 'auroc' or 'auprc' or 'youdenj' or 'f1' or 'precision' or 'recall' or 'seqacc') in args.eval_metrics:
            err = f'selected dataset (`{args.dataset}`) is for a regression task... please check evaluation metrics!'
            logger.exception(err)
            raise AssertionError(err)

    # print welcome message
    logger.info('[CONFIG] List up configurations...')
    for arg in vars(args):
        if 'glove_emb' in str(arg):
            if getattr(args, arg) is not None:
                logger.info(f'[CONFIG] - {str(arg).upper()}: USE!')
            else:
                logger.info(f'[CONFIG] - {str(arg).upper()}: NOT USE!')
            continue
        logger.info(f'[CONFIG] - {str(arg).upper()}: {getattr(args, arg)}')
    else:
        print('')
    return args

#####################
# BCEWithLogitsLoss #
#####################
class PainlessBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Native `torch.nn.BCEWithLogitsLoss` requires squeezed logits shape and targets with float dtype.
    """
    def __init__(self, **kwargs):
        super(PainlessBCEWithLogitsLoss, self).__init__(**kwargs)

    def forward(self, inputs, targets):
        return super(PainlessBCEWithLogitsLoss, self).forward(
            torch.atleast_1d(inputs.squeeze()), 
            torch.atleast_1d(targets).float()
        )

torch.nn.BCEWithLogitsLoss = PainlessBCEWithLogitsLoss

################
# Seq2Seq Loss #
################
class Seq2SeqLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(Seq2SeqLoss, self).__init__(**kwargs)

    def forward(self, inputs, targets, ignore_indices=torch.tensor([0, 1, 2, 3])):
        num_classes = inputs.size(-1)
        inputs, targets = inputs.view(-1, num_classes), targets.view(-1)
        targets[torch.isin(targets, ignore_indices.to(targets.device))] = -1
        return torch.nn.functional.cross_entropy(inputs, targets, ignore_index=-1)

torch.nn.Seq2SeqLoss = Seq2SeqLoss

##################
# Metric manager #
##################
class MetricManager:
    """Managing metrics to be used.
    """
    def __init__(self, eval_metrics):
        self.metric_funcs = {
            name: import_module(f'.metrics', package=__package__).__dict__[name.title()]()
            for name in eval_metrics
            }
        self.figures = defaultdict(int) 
        self._results = dict()

        # use optimal threshold (i.e., Youden's J or not)
        if 'youdenj' in self.metric_funcs:
            for func in self.metric_funcs.values():
                if hasattr(func, '_use_youdenj'):
                    setattr(func, '_use_youdenj', True)

    def track(self, loss, pred, true):
        # update running loss
        self.figures['loss'] += loss * len(pred)

        # update running metrics
        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step=None):
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        else:
            self._results = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
                }
        self.figures = defaultdict(int)

    @property
    def results(self):
        return self._results
