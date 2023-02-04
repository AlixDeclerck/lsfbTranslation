#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    learning_phase.py train --app-path=<file>
"""

import os

from common.constant import dir_separator
from docopt import docopt
from algorithms.data_loader.src.dal import EnvType
from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.tools.helper import load_config
from algorithms.symbolicTransformer.src.core.training import load_or_train_model


if __name__ == '__main__':

    # CONFIGURATION
    config = load_config()

    # update configuration of given parameters
    args = docopt(__doc__)
    config["application_path"] = os.environ['HOME'] + dir_separator + args['--app-path'] + dir_separator

    # PREPROCESSING
    vocab = Vocab(load_tokenizers(), config, EnvType.DEV)

    # TRAINING
    trained_model = load_or_train_model(vocab, config)

    # OUTPUT
    print(trained_model)
