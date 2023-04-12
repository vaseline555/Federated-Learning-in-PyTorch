import os
import copy
import torch
import random
import logging
import numpy as np
import concurrent.futures

from importlib import import_module
from collections import ChainMap, OrderedDict, defaultdict

from src import init_weights, TqdmToLogger
from .baseserver import BaseServer

logger = logging.getLogger(__name__)



class Server(BaseServer):
    """Centeral server orchestrating the whole process of federated learning.
    """
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        self.args = args
        self.writer = writer
        
        # round indicator
        self.round = 0

        # global holdout set
        if self.args.eval_type != 'local':
            self.server_dataset = server_dataset

        # model
        self.model = self._init_model(model)

        # clients
        self.clients = self._create_clients(client_datasets)

        # result container
        self.results = defaultdict(dict)

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; init type: {self.args.init_type.upper()})!')
        return model

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__['Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(self.args, *datasets)
            client.id, client.model = identifier, copy.deepcopy(self.model)
            return {identifier: client}

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                client_datasets.items(), 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)
                ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())            
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return dict(ChainMap(*clients)) 

    def _broadcast_models(self, ids):
        def __broadcast_model(client):
            client.download(self.model)
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Broadcast the global model at the server!')

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
            for identifier in TqdmToLogger(
                ids, 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...broadcasting server model... ',
                total=len(ids)
                ):
                workhorse.submit(__broadcast_model, self.clients[identifier]).result()
      
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully broadcasted the model to selected {len(ids)} clients!')

    def _sample_clients(self, exclude=[]):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Sample clients!')
        if exclude == []: # randomly select floor(C * K) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample(self.clients.keys(), num_sampled_clients))
        else: # randomly select unparticipated clients in amount of `eval_fraction` multiplied
            num_unparticipated_clients = int((1. - self.args.C) * self.args.K)
            num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
            sampled_client_ids = sorted(random.sample([identifier for identifier in self.clients.keys() if identifier not in exclude], num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    def _log_results(self, resulting_sizes, results, eval, participated):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDDATE"}] [CLIENT] < {str(identifier).zfill(8)} > '

            # get loss and metrics
            if eval:
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate intototal logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses = np.array(losses).astype(float)
        weighted = losses.dot(num_samples) / sum(num_samples)
        equal = losses.mean()
        std = losses.std()
        top = losses.max()
        total_log_string += f'\n    - Loss: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) | Top ({top:.4f})'
        result_dict['loss'] = {'weighted': weighted, 'equal': equal, 'std': std, 'top': top}
        if self.writer is not None:
            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
                {f'Weighted Average': weighted, f'Equal Average': equal, f'Top': top},
                self.round
            )

        # metrics
        for name, values in metrics.items():
            values = np.array(values).astype(float)
            weighted = values.dot(num_samples) / sum(num_samples)
            equal = values.mean()
            std = values.std()
            bottom = values.min()
            total_log_string += f'\n    - {name.title()}: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) | Bottom ({bottom:.4f})'
            result_dict[name] = {'weighted': weighted, 'equal': equal, 'std': std, 'bottom': bottom}
            if self.writer is not None:
                for name, values in metrics.items():
                    self.writer.add_scalars(
                        f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f'({"In" if participated else "Out"})',
                        {f'Weighted Average': weighted, f'Equal Average': equal, f'Bottom': bottom},
                        self.round
                    )
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval=False, participated=False):
        def __update_clients(client):
            self.args.lr *= self.args.lr_decay
            client.lr = self.args.lr
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client):
            eval_result = client.evaluate() 
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                    total=len(ids)
                    ):
                    results.append(workhorse.submit(__evaluate_clients, self.clients[idx]).result()) 
            eval_sizes, eval_results = list(map(list, zip(*results)))
            eval_sizes, eval_results = dict(ChainMap(*eval_sizes)), dict(ChainMap(*eval_results))
            self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                eval_sizes, 
                eval_results, 
                eval=True, 
                participated=participated
            )
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
        else:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() - 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...update clients... ',
                    total=len(ids)
                    ):
                    results.append(workhorse.submit(__update_clients, self.clients[idx]).result()) 
            update_sizes, update_results = list(map(list, zip(*results)))
            update_sizes, update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes, 
                update_results, 
                eval=False, 
                participated=True
            )
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes
    
    def _aggregate(self, ids, updated_sizes):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # call current server model and create empty container
        curr_state, new_state = self.model.state_dict(), OrderedDict()

        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: coefficient / sum(updated_sizes.values()) for identifier, coefficient in updated_sizes.items()}

        # aggregate weights
        for it, identifier in enumerate(ids):
            locally_updated_weights = self.clients[identifier].upload()
            for key in curr_state.keys():
                if it == 0:
                    new_state[key] = (1. - self.args.beta) * (coefficients[identifier] * locally_updated_weights[key])\
                        + self.args.beta * (curr_state[key])
                else:
                    new_state[key] += (1. - self.args.beta) * (coefficients[identifier] * locally_updated_weights[key])\
                        + self.args.beta * (curr_state[key])
        else: # update as a new server model
            self.model.load_state_dict(new_state)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')

    @torch.inference_mode()
    def _central_evaluate(self):
        #self.results[self.round]['server_evaluated'] = server_eval_result
        self.model.eval()
        self.model.to(self.args.device)

        losses, corrects = 0., 0.
        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            losses += len(outputs) * loss.item()
            corrects += (outputs.argmax(1) == targets).sum().item()
        else:
            epoch_loss, epoch_acc = losses / len(self.server_dataset), corrects / len(self.server_dataset)
            self.results[self.round]['server_evaluated']  = {'loss': epoch_loss, 'acc': epoch_acc}

    def update(self):
        """Update the global model through federated learning.
        """
        # randomly select clients
        selected_ids = self._sample_clients()

        # broadcast the current model at the server to selected clients
        self._broadcast_models(selected_ids)
        
        # request update to selected clients
        updated_sizes = self._request(selected_ids, eval=False)

        # request evaluation to selected clients
        self._request(selected_ids, eval=True, participated=True)

        # receive updates and aggregate into a new server model  
        self._aggregate(selected_ids, updated_sizes)
        return selected_ids

    def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        # randomly select all remaining clients not participated in current round
        selected_ids = self._sample_clients(exclude=excluded_ids)
        self._broadcast_models(selected_ids)

        # request evaluation 
        ## `local`: evaluate on selected clients' holdout set
        ## `global`: evaluate on the server's global holdout set 
        ## `both`: conduct both `local` and `global` evaluations
        if self.args.eval_type == 'local':
            self._request(selected_ids, eval=True, participated=False)
        elif self.args.eval_type == 'global':
            self._central_evaluate()
        elif self.args.eval_type == 'both':
            self._request(selected_ids, eval=True, participated=False)
            self._central_evaluate()

        # calculate generalization gap
        gen_gap = defaultdict(dict)
        curr_res = self.results[self.round]
        for key in ['loss', 'acc']:
            for name in curr_res['clients_evaluated_out'][key].keys():
                if ['equal', 'weighted'] in name:
                    gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                    gen_gap[f'gen_gap_{key}'][name] = gap
                    if self.writer is not None:
                        self.writer.add_scalars(
                            f'{key.title()} Generalization Gap',
                            {name: gap},
                            self.round
                        )
        else:
            self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save resulting figures and a trained model checkpoint. 
        """

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
        