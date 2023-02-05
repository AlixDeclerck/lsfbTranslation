# Symbolic Transformer

## Abstract
First step was to read the `progressive Transformer` (https://arxiv.org/abs/2004.14874) that show us a proven architecture to learn translation from common language to glosses.

Second step, basing on the `annotated transformer` last the state-of-the-art improvements (https://nlp.seas.harvard.edu/annotated-transformer/, https://nlp.seas.harvard.edu/2018/04/03/attention.html) we recreated `symbolic Transformer` that was the first step given in `progressive Transformer`

## Model architecture & pytorch

### distributed

A `distributed` boolean configure behaviors of the project for multiple gpu processing parallelization

- https://pytorch.org/tutorials/beginner/dist_overview.html

### dataloaders
A `label smoothing` using `The Kullback-Leibler divergence loss` criterion

- https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html

### dataloaders

The `torch.utils.data.DataLoader` is an iterator which provides features. 

- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

### optimizer

The optimizer implements `Adam algorithm`

- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

### scheduler and learning rate

- https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html







