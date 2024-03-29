# lsfbTranslation
Implementing and compare POS tagging and state-of-the-art model to generate sign language glosses wih NLP.

This `UMONS` project under the `PhD Dupont` direction is intended to translate French texts to glosses where glosses are a temporary step to generate visual signs. This is the Alix Declerck's Master 1 memory. 

This repository gather algorithms they are used to do NLP learning.

## Corpus

The corpus `Contes en gloses (LSF)` is 
- distributed via https://www.ortolang.fr
- a 3500 lines FR to LSF documents

## Installation documentation

https://conda.io/projects/conda/en/latest/commands.html

We run a python 3.9 environment with anaconda 22.9 using the following installation commands :

`conda info --envs & conda activate conda activate lsfbTranslation`

## Packages and libraries

### Data loader

- Reading CSV (initially for the Ben Sanders csv sources for learning)
- Handling database (populate, loadings, ...)

The `database creation folder` is destined to database sql creation scripts

```
python -m pip install mysql-connector-python
conda install -c anaconda openpyxl
```

### Other algorith

Local or general scripts quoted here only for libraries compatibility purpose

```
conda install -c conda-forge matplotlib
conda install -c conda-forge pycryptodome
conda install -c anaconda cryptography
conda install -c conda-forge scikit-plot
conda install -c conda-forge gensim
```

### Symbolic transformer

Based on `Progressive transformer` and `annotated transformer`
datasets : translated (from german with deepl) version of the `phoenix`

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
conda install -c conda-forge unidecode
conda install evaluate
conda install -c conda-forge fasttext
```

### Words are our glosses

seq2seq with meteo texts : French input, english output

```
conda install -c conda-forge docopt
conda install -c anaconda nltk
```

### Syntax analysis

POS tagging to generate glosses

```
conda install -c conda-forge stanza
conda install -c conda-forge textacy
```

## Credit and sources

Please see on each packages for details

@article{wordsAreOurGlosses,
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

@article{NMT_seq2seq_Standford,
author  = "Pencheng Yin, Sahil Chopra, Vera Lin",
title   = "Neural Machine Translation with sequence-to-sequence, attention, and subwords",
year    = "2019",
journal = "Standford"
}
