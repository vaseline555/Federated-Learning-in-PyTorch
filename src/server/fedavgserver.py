import os
import copy
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures

from importlib import import_module
from collections import ChainMap, defaultdict

from src import init_weights, TqdmToLogger
from .baseserver import BaseServer

logger = logging.getLogger(__name__)



class FedavgServer(BaseServer):
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

        # server aggregator
        self.server_optimizer = self._get_algorithm(self.model, lr=self.args.lr, momentum=self.args.beta)

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.server_optimizer, lr_lambda=lambda curr_round: self.args.lr_decay**curr_round)

        # clients
        self.clients = self._create_clients(client_datasets)

        # result container
        self.results = defaultdict(dict)

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; init type: {self.args.init_type.upper()})!')
        return model
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        return ALGORITHM_CLASS(params=model.parameters(), **kwargs)

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
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
            num_unparticipated_clients = max(int((1. - self.args.C) * self.args.K), 0)
            if num_unparticipated_clients == 0: # when C = 1; evaluate on all clients!
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted(self.clients.keys())
            else:
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
        for name, val in metrics.items():
            val = np.array(val).astype(float)
            weighted = val.dot(num_samples) / sum(num_samples)
            equal = val.mean()
            std = val.std()
            bottom = val.min()
            total_log_string += f'\n    - {name.title()}: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) | Bottom ({bottom:.4f})'
            result_dict[name] = {'weighted': weighted, 'equal': equal, 'std': std, 'bottom': bottom}
            if self.writer is not None:
                for name in metrics.keys():
                    self.writer.add_scalars(
                        f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f'({"In" if participated else "Out"})',
                        {f'Weighted Average': weighted, f'Equal Average': equal, f'Bottom': bottom},
                        self.round
                    )
                self.writer.flush()
        
        # log total message
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval=False, participated=False):
        def __update_clients(client):
            client.args.lr = self.lr_scheduler.get_last_lr()[-1]
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

        # empty out buffer
        self.server_optimizer.zero_grad()

        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: coefficient / sum(updated_sizes.values()) for identifier, coefficient in updated_sizes.items()}
        
        # accumulate weights
        for identifier in ids:
            locally_updated_weights_iterator = self.clients[identifier].upload()
            self.server_optimizer.accumulate(coefficients[identifier], locally_updated_weights_iterator)
        else:
            self.server_optimizer.step()
            self.lr_scheduler.step()
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')

    @torch.inference_mode()
    def _central_evaluate(self):
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
            result = {'loss': epoch_loss, 'metrics': {'accuracy': epoch_acc}}

            # log result
            server_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

            ## loss
            loss = result['loss']
            server_log_string += f'| loss: {loss:.4f} '
            
            ## metrics
            for metric, value in result['metrics'].items():
                server_log_string += f'| {metric}: {value:.4f} '
            logger.info(server_log_string)

            # log TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Server Loss', epoch_loss, self.round)
                for name, value in result['metrics'].items():
                    self.writer.add_scalar(f'Server {name.title()}', value, self.round)
                else:
                    self.writer.flush()
            self.results[self.round]['server_evaluated'] = result

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
        gen_gap = dict()
        curr_res = self.results[self.round]
        for key in curr_res['clients_evaluated_out'].keys():
            for name in curr_res['clients_evaluated_out'][key].keys():
                if name in ['equal', 'weighted']:
                    gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                    gen_gap[f'gen_gap_{key}'] = {name: gap}
                    if self.writer is not None:
                        self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
                        self.writer.flush()
        else:
            self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        # save figure
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file:
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)

        # save checkpoint
        torch.save(self.model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt'))
        
        if self.writer is not None:
            self.writer.close()
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
        