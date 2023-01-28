# lsfbTranslation
Implementing and compare state of the art model to generate sign language glosses wih NLP
This `umons` project under the PhD Dupont direction is intended to translate french texts to glosses where glosses are a temporary step to generate visual signs. This is the Alix Declerck's Master 1 end of studies project. This repository gather algorithms they are used to do NLP learning.

## Installation

### Documentation
https://conda.io/projects/conda/en/latest/commands.html

We run a python 3.9 environment with anaconda 22.9 using the following installation commands :
`conda info --envs & conda activate conda activate lsfbTranslation`



### Data loader

```
python -m pip install mysql-connector-python
conda install -c conda-forge pycryptodome
conda install -c anaconda cryptography
```

### Other algorith

```
conda install -c conda-forge matplotlib
```


### Progressive transformer

```
conda install -c conda-forge opencv-python (opencv)
./pip install torch-lr-scheduler
```

[a version stay's in signTranslation module]

```
conda install -c pytorch torchtext==0.08 (or ..)
conda install -c pytorch torchtext==0.11 (needed for import from torchtext.legacy.data import Dataset, Iterator, Field
```

### Symbolic transformer

```
conda install -c anaconda numpy
conda install -c anaconda pytorch
conda install -c conda-forge tensorflow
conda install -c conda-forge tensorflow-datasets
conda install -c conda-forge torchdata
conda install -c conda-forge spacy
conda install -c anaconda pandas
conda install -c conda-forge gputil
conda install -c pytorch torchtext==0.12
conda install -c anaconda altair
conda install -c anaconda pyyaml
+ installing package jupyter in intelliJ (pro)
```

### Words are our glosses

```
conda install -c conda-forge docopt
conda install -c anaconda nltk
```

### Syntax analysis

```
conda install -c conda-forge stanza
```


## Presentation

### Data loader
- Reading CSV (initially for the Ben Sanders csv sources for learning)
- Handling database (populate, loadings, ...)

#### DB Creation
The databases creation scripts

### Symbolic transformer AIAYN
Based on Progressive transformer
Adapted from http://nlp.seas.harvard.edu/annotated-transformer/

### Words are our glosses
seq2seq with meteo texts : French input, english output

### Syntax analysis
A graph that bring some prior knowledge that can be use as pre-processing ?
Visual Genome : http://visualgenome.org/

## Documentation links
Some useful documentations

- https://www.tensorflow.org/s/results/?q=vocab&hl=fr
- https://altair-viz.github.io/user_guide/customization.html
- https://github.com/rsennrich/subword-nmt
- https://github.com/OpenNMT/OpenNMT-py/
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- BPE : https://medium.com/@pierre_guillou/nlp-fastai-sentencepiece-d6922b5480d6
- CVS 2 JSON : https://csvjson.com/csv2json
- Image filtering with python : https://plainenglish.io/blog/image-filtering-and-editing-in-python-with-code-e878d2a4415d
- memory usage : https://www.kaggle.com/getting-started/140636
- code optimization : https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
- https://pytorch.org/text/0.8.1/datasets.html

## Credit and sources
Please see on each packages for details

@inproceedings{wordsAreOurGlosses,
author = {Zelinka, Jan and Kanis, Jakub},
year = {2020},
month = {03},
pages = {3384-3392},
title = {Neural Sign Language Synthesis: Words Are Our Glosses},
doi = {10.1109/WACV45572.2020.9093516}
}


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

@article{phoenix_dataset,
author  = {Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, Richard Bowden},
title   = "\url{https://paperswithcode.com/dataset/phoenix14t}",
year    = "2018",
journal = "web"
}


@article{the_annotated_transformer,
author  = {Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman},
title   = "\url{http://nlp.seas.harvard.edu/annotated-transformer/}",
year    = "2022",
journal = "web"
}