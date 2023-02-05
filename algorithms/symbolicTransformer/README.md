# Symbolic Transformer

## Abstract
First step was to read the `progressive Transformer` (https://arxiv.org/abs/2004.14874) that show us a proven architecture to learn translation from common language to glosses.

Second step, basing on the `annotated transformer` last the state-of-the-art improvements (https://nlp.seas.harvard.edu/annotated-transformer/, https://nlp.seas.harvard.edu/2018/04/03/attention.html) we recreated `symbolic Transformer` that was the first step given in `progressive Transformer`

## Model architecture & pytorch

### distributed

A `distributed` boolean configure behaviors of the project, seam to be a gpu processing parallelization

The DataParallel package enables single-machine multi-GPU parallelism with the lowest coding hurdle.

- https://pytorch.org/tutorials/beginner/dist_overview.html

### dataloaders

torch.utils.data.DataLoader is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. However, default collate should work fine for most use cases.

- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

### optimizer

- https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

### scheduler and learning rate

- https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html







