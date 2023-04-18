import os
import time
import torch
import argparse

from importlib import import_module
from torch.utils.tensorboard import SummaryWriter

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model 



def main(args, writer):
    """Main program to run federated learning.
    
    Args:
        args: user input arguments parsed by argparser
        writer: `torch.utils.tensorboard.SummaryWriter` instance for TensorBoard tracking
    """
    # set seed for reproducibility
    set_seed(args.seed)
    
    # get dataset
    server_dataset, client_datasets = load_dataset(args)
    
    # adjust device
    if 'cuda' in args.device:
        assert torch.cuda.is_available(), 'Please check if your GPU is available now!' 
        args.device = 'cuda' if args.device_ids == [] else f'cuda:{args.device_ids[0]}'
    
    # get model
    model, args = load_model(args)

    # create central server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}Server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model=model)
    
    # federated learning
    for curr_round in range(1, args.R + 1):
        ## update round indicator
        server.round = curr_round

        ## update after sampling clients randomly
        selected_ids = server.update() 

        ## evaluate on clients not sampled (for measuring generalization performance)
        if curr_round % args.eval_every == 0:
            server.evaluate(excluded_ids=selected_ids)
    else:
        ## wrap-up
        server.finalize()

    

if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    #####################
    # Default arguments #
    #####################
    parser.add_argument('--exp_name', help='experiment name', type=str, required=True)
    parser.add_argument('--seed', help='global random seed', type=int, default=5959)
    parser.add_argument('--device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default='cpu')
    parser.add_argument('--device_ids',  nargs='+', type=int, help='GPU device ids for multi-GPU training (use all available GPUs if no number is passed)', default=[])
    parser.add_argument('--data_path', help='path to read data from', type=str, default='./data')
    parser.add_argument('--log_path', help='path to store logs', type=str, default='./log')
    parser.add_argument('--result_path', help='path to save results', type=str, default='./result')
    parser.add_argument('--use_tb', help='use TensorBoard to track logs (if passed)', action='store_true')
    parser.add_argument('--tb_port', help='TensorBoard port number (valid only if `use_tb`)', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address (valid only if `use_tb`)', type=str, default='0.0.0.0')
    
    #####################
    # Dataset arguments #
    #####################
    ## dataset
    parser.add_argument('--dataset', help='''name of dataset to use for an experiment 
    * NOTE: case sensitive*
    - image classification datasets in `torchvision.datasets`,
    - text classification datasets in `torchtext.datasets`,
    - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
    - among [ TinyImageNet | CINIC10 | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
    ''', type=str, required=True)
    parser.add_argument('--test_fraction', help='fraction of hold-out dataset for evaluation', type=float, choices=[Range(0., 1.)], default=0.2)

    ## data augmentation arguments
    parser.add_argument('--resize', help='resize input images (using `torchvision.transforms.Resize`)', type=int, default=28)
    parser.add_argument('--imnorm', help='normalize channels using ImageNet pre-trained mean & standard deviation (using `torchvision.transforms.Normalize`)', action='store_true')
    parser.add_argument('--randrot', help='randomly rotate input (using `torchvision.transforms.RandomRotation`)', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip input horizontaly (using `torchvision.transforms.RandomHorizontalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randvf', help='randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    
    ## statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''type of data split scenario
    - `iid`: statistically homogeneous setting,
    - `unbalanced`: unbalance in sample counts across clients,
    - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
    - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
    - `pre`: pre-defined data split scenario
    ''', type=str, choices=['iid', 'unbalanced', 'patho', 'diri', 'pre'], required=True)
    parser.add_argument('--mincls', help='the minimum number of distinct classes per client (only used when `split_type` is `patho`)', type=int, default=2)
    parser.add_argument('--cncntrtn', help='a concentration parameter for Dirichlet distribution (only used when `split_type` is `diri`)', type=float, default=0.1)
    parser.add_argument('--rawsmpl', help='fraction of raw data to be used (only used when one of `LEAF` datasets is used)', type=float, choices=[Range(0., 1.)], default=1.0)
    
    ###################
    # Model arguments #
    ###################
    ## model
    parser.add_argument('--model_name', help='a model to be used (note that it is case sensitive)', type=str,
        choices=[
            'TwoNN', 'TwoCNN',
            'LeNet', 'MobileNet', 'SqueezeNet',
            'VGG9', 'VGG9BN', 'VGG11', 'VGG11BN', 'VGG13', 'VGG13BN',
            'ResNet10', 'ResNet18', 'ResNet34',
            'MobileNeXt', 'SqueezeNeXt', 'MobileViT', 
            'NextCharLSTM', 'NextWordLSTM',
            'DistilBert', 'SqueezeBert', 'MobileBert',
            'LogReg', 'GRUClassifier'
        ],
        required=True
    )
    parser.add_argument('--hidden_size', help='hidden channel size for vision models, or hidden dimension of language models', type=int, default=64)
    parser.add_argument('--dropout', help='dropout rate', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--use_model_tokenizer', help='use a model-specific tokenizer (if passed)', action='store_true')
    parser.add_argument('--use_pt_model', help='use a pre-trained model weights for fine-tuning (if passed)', action='store_true')
    parser.add_argument('--seq_len', help='maximum sequence length used for `torchtext.datasets`)', type=int, default=512)
    parser.add_argument('--num_layers', help='number of layers in recurrent cells', type=int, default=2)
    parser.add_argument('--num_embeddings', help='size of embedding dictionary', type=int, default=1000)
    parser.add_argument('--embedding_size', help='embedding dimension of language models', type=int, default=512)
    parser.add_argument('--init_type', help='weight initialization method', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'])
    parser.add_argument('--init_gain', type=float, default=1.0, help='magnitude of variance used for weight initialization')
    
    ######################
    # Learning arguments #
    ######################
    ## federated learning settings
    parser.add_argument('--algorithm', help='type of an federated learning algorithm to be used', type=str,
        choices=['fedavg', 'fedsgd', 'fedprox'], 
        required=True
    )
    parser.add_argument('--eval_type', help='''the evaluation type of a model trained from FL algorithm
    - `local`: evaluation of personalization model on local hold-out dataset  (i.e., evaluate personalized models using each client\'s local evaluation set)
    - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the global model using separate holdout dataset located at the server)
    - 'both': combination of `local` and `global` setting
    ''', type=str,
        choices=['local', 'global', 'both'],
        required=True
    )
    parser.add_argument('--eval_fraction', help='fraction of randomly selected clients for evaluation (for `eval_type` is `local` or `both`)', type=float, choices=[Range(1e-8, 1.)], default=1.)
    parser.add_argument('--eval_every', help='frequency of the evaluation (i.e., evaluate peformance of a model every `eval_every` round)', type=int, default=1)
    parser.add_argument('--C', help='sampling fraction of clietns per round (full participation when zero is passed)', type=float, choices=[Range(1e-8, 1.)], default=0.1)
    parser.add_argument('--K', help='number of total cilents participating in federated training', type=int, default=100)
    parser.add_argument('--R', help='number of total rounds', type=int, default=5000)
    parser.add_argument('--E', help='number of local epochs', type=int, default=5)
    parser.add_argument('--B', help='batch size for local update in each client (full-batch training when zero is passed)', type=int, default=10)
    parser.add_argument('--beta', help='global momentum factor for an update of a global model when aggregated at the server', type=float, choices=[Range(0., 1.)], default=0.)
    
    # optimization arguments
    parser.add_argument('--no_shuffle', help='do not shuffle data (if passed)', action='store_true')
    parser.add_argument('--optimizer', help='type of optimization method (should be a sub-module of `torch.optim`)', type=str, default='SGD')
    parser.add_argument('--weight_decay', help='weight decay (L2 penalty)', type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--momentum', help='momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--lr', help='learning rate for local updates in each client', type=float, choices=[Range(0., 100.)], default=0.01)
    parser.add_argument('--lr_decay', help='rate of learning rate decay applied per round', type=float, choices=[Range(0., 1.)], default=0.9999)
    parser.add_argument('--criterion', help='type of criterion for objective function (should be a submodule of `torch.nn`)', type=str, default='CrossEntropyLoss')
    parser.add_argument('--mu', help='constant for proximity regularization term (for algorithms `fedprox`)', type=float, choices=[Range(0., 100)], default=0.01)

    # parse arguments
    args = parser.parse_args()
    
    # check arguments
    args = check_args(args)

    # make path for saving losses & metrics & models
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.exp_name = f'{args.exp_name}_{args.seed}_{args.dataset.lower()}_{args.model_name.lower()}'
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    # make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    # initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)
    
    # run main program
    if args.use_tb: # run TensorBaord for tracking losses and metrics
        writer = SummaryWriter(
            log_dir=os.path.join(args.log_path, f'{args.exp_name}_{curr_time}'), 
            filename_suffix=f'_{curr_time}'
        )
        tb = TensorBoardRunner(os.path.join(args.log_path, f'{args.exp_name}_{curr_time}'), args.tb_host, args.tb_port)
        main(args, writer)
        tb.finalize()
    else:
        main(args, None)
    
    # bye!
    os._exit(0)
