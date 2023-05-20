# Symbolic Transformer

## Abstract
What we call the symbolic transformer is inspired by the first architecture of the `progressive Transformer` (see credit).
We input texts and output glosses using `pytorch`, `spacy` and `fast-text` though the `annotated transformer` (see credit).
We make it more runnable using the `standford NMT` (see credit) and some functionalities.

## Model architecture & pytorch

### distributed

A `distributed` boolean configure behaviors of the project for multiple gpu processing parallelization

- https://pytorch.org/tutorials/beginner/dist_overview.html

### label smoothing
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

## Credits

@article{progressiveTransformer,
doi = {10.48550/ARXIV.2004.14874},
url = {https://arxiv.org/abs/2004.14874},
author = {Saunders, Ben and Camgoz, Necati Cihan and Bowden, Richard},
keywords = {Computer Vision and Pattern Recognition (cs.CV), Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
title = {Progressive Transformers for End-to-End Sign Language Production},
publisher = {arXiv},
year = {2020},
copyright = {arXiv.org perpetual, non-exclusive license}
}

@article{the_annotated_transformer,
author  = {Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman},
title   = "\url{http://nlp.seas.harvard.edu/annotated-transformer/}",
year    = "2022",
journal = "web"
}

@article{NMT_seq2seq_Standford,
author  = "Pencheng Yin, Sahil Chopra, Vera Lin",
title   = "Neural Machine Translation with sequence-to-sequence, attention, and subwords",
year    = "2019",
journal = "Standford"
}