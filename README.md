
# Federated Learning in PyTorch
Implementations of various Federated Learning (FL) algorithms in PyTorch, especially for research purpose.

## Implementation Details
### Datasets
* Supports all image classification datasets in `torchvision.datasets`.
* Supports all text classification datasets in `torchtext.datasets`.
* Supports all datasets in [LEAF benchmark](https://leaf.cmu.edu/) (*NO need to prepare raw data manually*)
* Supports additional image classification datasets ([`TinyImageNet`](https://www.kaggle.com/c/tiny-imagenet), [`CINIC10`](https://datashare.ed.ac.uk/handle/10283/3192)).
* Supports additional text classification datasets ([`BeerReviews`](https://snap.stanford.edu/data/web-BeerAdvocate.html)).
* Supports tabular datasets ([`Heart`, `Adult`, `Cover`](https://archive.ics.uci.edu/ml/index.php)).
* Supports temporal dataset ([`GLEAM`](http://www.skleinberg.org/data.html))
* __NOTE__: don't bother to search raw files of datasets; dataset can automatically be downloaded to the designated path by just passing its name!
### Statistical Heterogeneity Simulations
* `IID` (i.e., statistical homogeneity)
* `Unbalanced` (i.e., sample counts heterogeneity)
* `Pathological Non-IID` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `Dirichlet distribution-based Non-IID` ([Hsu et al., 2019](https://arxiv.org/abs/1909.06335))
* `Pre-defined` (for datasets having natural semantic separation, including `LEAF` benchmark ([Caldas et al., 2018](https://arxiv.org/abs/1812.01097)))
### Models
* `LogReg` (logistic regression), `GRUClassifier` (GRU-cell based classifier)
* `TwoNN`, `TwoCNN`, `NextCharLSTM`, `NextWordLSTM` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `LeNet` ([LeCun et al., 1998](https://ieeexplore.ieee.org/document/726791/)), `MobileNet` ([Howard et al., 2019](https://arxiv.org/abs/1905.02244)), `SqueezeNet` ([Iandola et al., 2016](https://arxiv.org/abs/1602.07360)), `VGG` ([Simonyan et al., 2014](https://arxiv.org/abs/1409.1556)), `ResNet` ([He et al., 2015](https://arxiv.org/abs/1512.03385))
* `MobileNeXt` ([Daquan et al., 2020](https://arxiv.org/abs/2007.02269)), `SqueezeNeXt` ([Gholami et al., 2016](https://arxiv.org/abs/1803.10615)), `MobileViT` ([Mehta et al., 2021](https://arxiv.org/abs/2110.02178))
* `DistilBERT` ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)), `SqueezeBERT` ([Iandola et al., 2020](https://arxiv.org/abs/2006.11316)), `MobileBERT` ([Sun et al., 2020](https://arxiv.org/abs/2004.02984))
### Algorithms
* `FedAvg` and `FedSGD` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `FedProx` ([Li et al., 2018](https://arxiv.org/abs/1812.06127))
### Evaluation schemes
* `local`: evaluate FL algorithm using holdout set of (some/all) clients NOT participating in the current round. (i.e., evaluation of personalized federated learning setting)
* `global`: evaluate FL algorithm using global holdout set located at the server. (*ONLY available if the raw dataset supports pre-defined validation/test set*).
* `both`: evaluate FL algorithm using both `local` and `global` schemes.
### Metrics
* Top-1 Accuracy, Top-5 Accuracy, Precision, Recall, F1
* Area under ROC, Area under PRC, Youden's J
* Seq2Seq Accuracy
* MSE, RMSE, MAE, MAPE
* $R^2$, $D^2$

## Requirements
* See `requirements.txt`. (I recommend to build an independent environment for this project, using e.g., `Docker` or `conda`)
* When you install `torchtext`, please check the version compatibility with `torch`. (See [official installation guide](https://github.com/pytorch/text#installation))
* Plus, please install `torch`-related packages using one command provided by the official guide (See [official repository](https://pytorch.org/get-started/locally/)); e.g., `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge` 

## Configurations
* See `python3 main.py -h`.

## Example Commands
* See shell files prepared in `commands` directory.

## TODO
- [ ] Support another models, especially lightweight ones for cross-device FL setting. (e.g., [`EdgeNeXt`](https://github.com/mmaaz60/EdgeNeXt))
- [ ] Support another structured datasets including temporal and tabular data, along with datasets suitable for cross-silo FL setting. (e.g., [`MedMNIST`](https://github.com/MedMNIST/MedMNIST))
- [ ] Add other popular FL algorithms including personalized FL algorithms (e.g., [`SuPerFed`](https://arxiv.org/abs/2109.07628)).
- [ ] Attach benchmark results of sample commands.

## Contact
Should you have any feedback, please create a thread in __issue__ tab, or contact me via `sjhahn11512@gmail.com`. Thank you :)
