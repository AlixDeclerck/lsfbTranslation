#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    learning_phase.py train --app-path=<file>
"""

import os

from docopt import docopt

from algorithms.symbolicTransformer.src.core.training import launch
from algorithms.symbolicTransformer.src.functionnal.data_preparation import Vocab
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config

if __name__ == '__main__':

    # CONFIGURATION
    config = load_config()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2400"

    # update configuration of given parameters
    args = docopt(__doc__)
    config["configuration_path"]["application_path"] = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"]

    # Retrieve vocabulary from disk
    vocab = Vocab(config)
    vocab.retrieve_from_disk()

    # TRAINING
    trained_model = launch(vocab, config)

    print("Symbolic transformer's training completed !!")
