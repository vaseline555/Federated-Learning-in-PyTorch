* NOTE: This repository will be updated to *ver 2.0* at least in August, 2022.
# Federated Averaging (FedAvg) in PyTorch [![arXiv](https://img.shields.io/badge/arXiv-1602.05629-f9f107.svg)](https://arxiv.org/abs/1602.05629)

An unofficial implementation of `FederatedAveraging` (or `FedAvg`) algorithm proposed in the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) in PyTorch. (implemented in Python 3.9.2.)

## Implementation points
* Exactly implement the models ('2NN' and 'CNN' mentioned in the paper) to have the same number of parameters written in the paper.
  * 2NN: `TwoNN` class in `models.py`; 199,210 parameters
  * CNN: `CNN` class in `models.py`; 1,663,370 parameters
* Exactly implement the non-IID data split.
  * Each client has at least two digits in case of using `MNIST` dataset.
* Implement multiprocessing of _client update_ and _client evaluation_.
* Support TensorBoard for log tracking.

## Requirements
* See `requirements.txt`
* When you install `torchtext`, please check compatibility with `torch` (see https://github.com/pytorch/text#installation)
* Plus, please install `torch`-related packages  via one command; e.g., `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge`

## Configurations
* See `python3 main.py -h`

## TODO
- [ ] More experiments with other hyperparameter settings (e.g., different combinations of B, E, K, and C)
- [ ] Support strucuted datasets suitable for FL from LibSVM, UCI ML repository, KDD cup, etc.
- [ ] FedSGD support