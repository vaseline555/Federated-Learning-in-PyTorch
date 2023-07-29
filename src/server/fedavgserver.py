import os
import gc
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures

from importlib import import_module
from collections import ChainMap, defaultdict

from src import init_weights, TqdmToLogger, MetricManager
from .baseserver import BaseServer

logger = logging.getLogger(__name__)



class FedavgServer(BaseServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        super(FedavgServer, self).__init__()
        self.args = args
        self.writer = writer

        self.round = 0 # round indicator
        if self.args.eval_type != 'local': # global holdout set for central evaluation
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model) # global model
        self.server_optimizer = self._get_algorithm(self.global_model, lr=self.args.lr, momentum=self.args.beta) # aggregation algorithm
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.server_optimizer, gamma=self.args.lr_decay) # global learning rate scheduler
        self.clients = self._create_clients(client_datasets) # clients container
        self.results = defaultdict(dict) # logging results container

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        return model
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        return ALGORITHM_CLASS(params=model.parameters(), **kwargs)

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)
                ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())            
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients

    def _broadcast_models(self, ids):
        def __broadcast_model(client):
            client.download(self.global_model)
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Broadcast the global model at the server!')
        self.global_model.to('cpu')
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
        if exclude == []: # Update - randomly select max(floor(C * K), 1) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))
        else: # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    def _log_results(self, resulting_sizes, results, eval, participated):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
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

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses = np.array(losses).astype(float)
        weighted = losses.dot(num_samples) / sum(num_samples); equal = losses.mean(); std = losses.std(); top10p = np.percentile(losses, 90)
        total_log_string += f'\n    - Loss: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) | Top 10% ({top10p:.4f})'
        result_dict['loss'] = {'weighted': weighted, 'equal': equal, 'std': std, 'top10%': top10p}
        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
            {'Weighted Average': weighted, 'Equal Average': equal, 'Std.': std, 'Top 10%': top10p},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val = np.array(val).astype(float)
            weighted = val.dot(num_samples) / sum(num_samples); equal = val.mean(); std = val.std(); bot10p = np.percentile(val, 10)
            total_log_string += f'\n    - {name.title()}: Weighted Avg. ({weighted:.4f}) | Equal Avg. ({equal:.4f}) | Std. ({std:.4f}) | Bottom 10% ({bot10p:.4f})'
            result_dict[name] = {'weighted': weighted, 'equal': equal, 'std': std, 'bottom10%': bot10p}
            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
                {'Weighted Average': weighted, 'Equal Average': equal, 'Std.': std, 'Bottom 10%': bot10p},
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
            if client.model is None:
                client.download(self.global_model)
            eval_result = client.evaluate() 
            client.model = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args._train_only:
                return
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
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}
        
        # accumulate weights
        for identifier in ids:
            locally_updated_weights_iterator = self.clients[identifier].upload()
            self.server_optimizer.accumulate(coefficients[identifier], locally_updated_weights_iterator)
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')

    @torch.inference_mode()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.eval()
        self.global_model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.global_model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log TensorBoard
        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        else:
            self.writer.flush()
        self.results[self.round]['server_evaluated'] = result

    def update(self):
        """Update the global model through federated learning.
        """
        #################
        # Client Update #
        #################
        selected_ids = self._sample_clients() # randomly select clients
        self._broadcast_models(selected_ids) # broadcast the current model at the server to selected clients
        updated_sizes = self._request(selected_ids, eval=False) # request update to selected clients
    
        #################
        # Server Update #
        #################
        self.server_optimizer.zero_grad() # empty out buffer
        self._aggregate(selected_ids, updated_sizes) # aggregate local updates
        self.server_optimizer.step() # update global model with by the aggregated update
        if self.round % self.args.lr_decay_step == 0: # update local learning rate
            self.lr_scheduler.step()
        self._request(selected_ids, eval=True, participated=True) # request evaluation to selected clients 
        # (this is located at last INTENTIONALLY to free out model pointer without problem)
        return selected_ids

    def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        ##############
        # Evaluation #
        ##############
        if self.args.eval_type != 'global': # `local` or `both`: evaluate on selected clients' holdout set
            selected_ids = self._sample_clients(exclude=excluded_ids)
            self._request(selected_ids, eval=True, participated=False)
        if self.args.eval_type != 'local': # `global` or `both`: evaluate on the server's global holdout set 
            self._central_evaluate()

        # calculate generalization gap
        if (not self.args._train_only) and (not self.args.eval_type == 'global'):
            gen_gap = dict()
            curr_res = self.results[self.round]
            for key in curr_res['clients_evaluated_out'].keys():
                for name in curr_res['clients_evaluated_out'][key].keys():
                    if name in ['equal', 'weighted']:
                        gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                        gen_gap[f'gen_gap_{key}'] = {name: gap}
                        self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
                        self.writer.flush()
            else:
                self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file: # save results
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)
        torch.save(self.global_model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt')) # save model checkpoint
        self.writer.close()
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
        if self.args.use_tb:
            input('[FINISH] ...press <Enter> to exit after tidying up your TensorBoard logging!')
