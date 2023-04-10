import ray
import copy
import torch
import logging

import numpy as np

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from collections import ChainMap, defaultdict

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
        self._broadcast([i for i in range(self.args.K)])
        self._round += 1

        # result container
        self.results = defaultdict(dict)

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Initialize a model!')
        model = init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; init type: {self.args.init_type.upper()})!')
        return model

    def _create_clients(self, client_datasets):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Create clients!')
        clients = []
        for k in trange(self.args.K, leave=False):
            clients.append(Client(self.args, k, *client_datasets[k]))
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients 

    def _broadcast(self, indices):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Broadcast the server model!')
        for idx in tqdm(indices, leave=False):
            self.clients[idx].model = copy.deepcopy(self.model)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...sucessfully broadcasted to selected {len(indices)} clients!')

    def _sample(self):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Sample clients!')
        num_sampled_clients = max(int(self.args.C * self.args.K), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.args.K)], size=num_sampled_clients, replace=False).tolist())
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_indices

    def _request(self, indices, eval=False):
        @ray.remote
        def update_clients(client):
            update_result = client.update()
            return {client.identifier: len(client)}, {client.identifier: update_result}

        @ray.remote
        def evaluate_clients(client):
            eval_result = client.evaluate() 
            return {client.identifier: eval_result}

        def log_results(results, eval=False):
            for identifier, result in results.items():
                if eval:
                    logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] [EVALUATE] [CLIENT] < {str(identifier).zfill(8)} > | Loss: {result["loss"]:.4f} | Acc.: {result["acc"]:.4f}')
                else:
                    logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] [UPDATE] [CLIENT] < {str(identifier).zfill(8)} > | Loss: {result[self.args.E]["loss"]:.4f} | Acc.: {result[self.args.E]["acc"]:.4f}') 

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if indices is None else len(indices)} clients!')
        if eval:
            work_ids = [evaluate_clients.remote(self.clients[idx]) for idx in indices]
            with logging_redirect_tqdm():
                eval_results = dict(ChainMap(*[eval_result for eval_result in tqdm(to_iterator(work_ids), total=len(indices), leave=False)]))
            log_results(eval_results, eval=True)
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed evaluation of {"all" if indices is None else len(indices)} clients!')
            return eval_results
        else:
            work_ids = [update_clients.remote(self.clients[idx]) for idx in indices]
            with logging_redirect_tqdm():
                results = [(length, update_result) for length, update_result in tqdm(to_iterator(work_ids), total=len(indices), leave=False)]
            update_sizes, update_results = list(map(list, zip(*results)))
            update_sizes, update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*update_results))
            log_results(update_results, eval=False)
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...completed updates of {"all" if indices is None else len(indices)} clients!')
            return update_sizes, update_results
    
    def _aggregate(self, indices, update_sizes):
        pass

    def _evaluate(self):
        pass

    def update(self):
        indices = self._sample()
        self._broadcast(indices)
        update_sizes, update_results = self._request(indices, eval=False)
        self.results[self._round]['clients_updated'] = update_results
        self._aggregate(indices, update_sizes)
        return indices

    def evaluate(self, indices):
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

    '''
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Start updating selected client {st[{self.args.algorithm.upper()}] r(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        self.clients[selected_index].client_update()
        client_size = len(self.clients[selected_index])

        message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...client {st[{self.args.algorithm.upper()}] r(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size

    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

        message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        if self.mp_flag:
            message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
            print(message); logging.info(message)
            del message; gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        return test_loss, test_accuracy

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            test_loss, test_accuracy = self.evaluate_global_model()
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Accuracy', 
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
                self._round
                )

            message = f"[{self.args.algorithm.upper()}] [Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()'''
