#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    learning_phase.py train --app-path=<file>
"""

import os

from common.constant import dir_separator
from docopt import docopt
from common.constant import EnvType
from algorithms.symbolicTransformer.src.functionnal.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from algorithms.symbolicTransformer.src.core.training import load_or_train_model


if __name__ == '__main__':

    # CONFIGURATION
    config = load_config()

    # update configuration of given parameters
    args = docopt(__doc__)
    config["application_path"] = os.environ['HOME'] + dir_separator + args['--app-path'] + dir_separator

    # PREPROCESSING
    vocab = Vocab(load_tokenizers(), config)

    # TRAINING
    trained_model = load_or_train_model(vocab, EnvType.TRAINING.value, config)

    # OUTPUT
    print(trained_model)
