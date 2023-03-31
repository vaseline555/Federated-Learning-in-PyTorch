import torch
import inspect
import logging
import importlib

logger = logging.getLogger(__name__)



def load_model(args):
    # retrieve model skeleton
    model_class = importlib.import_module('..models', package=__package__).__dict__[args.model_name]

    # get required model arguments
    required_args = inspect.getargspec(model_class)[0]

    # collect eneterd model arguments
    model_args = {}
    for argument in required_args:
        if argument == 'self': 
            continue
        model_args[argument] = getattr(args, argument)

    # get model instance
    model = model_class(**model_args)

    # adjust arguments if needed
    if args.use_pt_model:
        args.num_embeddings = model.num_embeddings
        args.embedding_size = model.embedding_size
        args.num_hiddens = model.num_hiddens
        args.dropout = model.dropout
    
    if args.model_name == 'LogReg':
        args.criterion = torch.nn.NLLLoss
    return model, args
