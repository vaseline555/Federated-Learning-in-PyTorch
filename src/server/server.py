import os
import ray
import copy
import torch
import logging

import numpy as np

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from collections import ChainMap, defaultdict
from multiprocessing import pool

from src import init_weights, to_iterator
from src.client import Client

logger = logging.getLogger(__name__)



class Server:
    """Centeral server orchestrating the whole process of federated learning.
    """
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        self.args = args
        self.writer = writer

        # current round indicator
        self._round = 0

        # global holdout set
        if self.args.eval_type != 'local':
            self.server_dataset = server_dataset

        # model
        self.model = self._init_model(model)

        # clients
        self.clients = self._create_clients(client_datasets)
        self._broadcast_models([i for i in range(self.args.K)])
        self._round += 1

        # result container
        self.results = defaultdict(dict)

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Initialize a model!')
        model = init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; init type: {self.args.init_type.upper()})!')
        return model

    def _create_clients(self, client_datasets):
        def __create_client(idx):
            return Client(self.args, idx, *client_datasets[idx])

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Create clients!')
        with pool.ThreadPool(processes=os.cpu_count() - 1) as workhorse:
            with logging_redirect_tqdm():
                clients = workhorse.map(__create_client, [k for k in tqdm(range(self.args.K), leave=False)])
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients 

    def _broadcast_models(self, indices):
        def __broadcast_model(client):
            client.model = copy.deepcopy(self.model)

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Broadcast the server model!')
        with pool.ThreadPool(processes=os.cpu_count() - 1) as workhorse:
            with logging_redirect_tqdm():
                _ = workhorse.map(__broadcast_model, [self.clients[k] for k in tqdm(indices, leave=False)])
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully broadcasted to selected {len(indices)} clients!')

    def _sample_clients(self):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Sample clients!')
        num_sampled_clients = max(int(self.args.C * self.args.K), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.args.K)], size=num_sampled_clients, replace=False).tolist())
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_indices

    def _request(self, indices, eval=False):
        @ray.remote
        def __update_clients(client):
            update_result = client.update()
            return {client.identifier: len(client)}, {client.identifier: update_result}

        @ray.remote
        def __evaluate_clients(client):
            eval_result = client.evaluate() 
            return {client.identifier: eval_result}

        def __log_results(results, eval=False):
            for identifier, result in results.items():
                if eval:
                    logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] [EVALUATE] [CLIENT] < {str(identifier).zfill(8)} > | Loss: {result["loss"]:.4f} | Acc.: {result["acc"]:.4f}')
                else:
                    logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] [UPDATE] [CLIENT] < {str(identifier).zfill(8)} > | Loss: {result[self.args.E]["loss"]:.4f} | Acc.: {result[self.args.E]["acc"]:.4f}') 

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if indices is None else len(indices)} clients!')
        if eval:
            work_ids = [__evaluate_clients.remote(self.clients[idx]) for idx in indices]
            with logging_redirect_tqdm():
                eval_results = dict(ChainMap(*[eval_result for eval_result in tqdm(to_iterator(work_ids), total=len(indices), leave=False)]))
            __log_results(eval_results, eval=True)
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed evaluation of {"all" if indices is None else len(indices)} clients!')
            return eval_results
        else:
            work_ids = [__update_clients.remote(self.clients[idx]) for idx in indices]
            with logging_redirect_tqdm():
                results = [(length, update_result) for length, update_result in tqdm(to_iterator(work_ids), total=len(indices), leave=False)]
            update_sizes, update_results = list(map(list, zip(*results)))
            update_sizes, update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*update_results))
            __log_results(update_results, eval=False)
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed updates of {"all" if indices is None else len(indices)} clients!')
            return update_sizes, update_results
    
    def _aggregate(self, indices, update_sizes):
        pass

    def _evaluate(self):
        pass

    def update(self):
        indices = self._sample_clients()
        self._broadcast_models(indices)
        update_sizes, update_results = self._request(indices, eval=False)
        self.results[self._round]['clients_updated'] = update_results
        self._aggregate(indices, update_sizes)

    def evaluate(self):
        indices = self._sample_clients()
        if self.args.eval_type == 'local':
            clients_eval_results = self._request(indices, eval=True)
            self.results[self._round]['clients_evaluated'] = clients_eval_results
        elif self.args.eval_type == 'global':
            server_eval_result = self._evaluate()
            self.results[self._round]['server_evaluated'] = server_eval_result
        elif self.args.eval_type == 'both':
            clients_eval_results = self._request(indices, eval=True)
            server_eval_result = self._evaluate()
            self.results[self._round]['clients_evaluated'] = clients_eval_results
            self.results[self._round]['server_evaluated'] = server_eval_result

    def finalize(self):
        """ 아래 내용 전부 server 내부 기능으로 삽입
        with open(os.path.join(args.result_path, f'{args.exp_name}_results.pkl'), 'wb') as result_file:
            arguments = {'arguments': {str(arg): getattr(args, arg) for arg in vars(args)}}
            results = {'results': {key: value for key, value in central_server.results.items() if len(value) > 0}}
            json.dump({**arguments, **results}, result_file, indent=4)
        
        # save checkpoints
        checkpoint = server.global_model.state_dict()

        # save checkpoints
        torch.save(checkpoint, os.path.join(args.result_path, f'{args.exp_name}_ckpt.pt'))
        """
        if self.writer is not None:
            self.writer.close()
        