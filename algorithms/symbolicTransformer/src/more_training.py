#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    learning_phase.py train --app-path=<file>
"""

import os
import torch

from common.constant import dir_separator
from docopt import docopt
from algorithms.data_loader.src.dal import EnvType
from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.tools.helper import load_config
from algorithms.symbolicTransformer.src.core.training import train_worker


if __name__ == '__main__':

    # CONFIGURATION
    config = load_config()
    torch.cuda.empty_cache()

    # RuntimeError: CUDA error: out of memory
    # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
    # For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")

    # update configuration of given parameters
    args = docopt(__doc__)
    config["application_path"] = os.environ['HOME'] + dir_separator + args['--app-path'] + dir_separator

    # PREPROCESSING
    vocab = Vocab(load_tokenizers(), config, EnvType.DEV)

    # TRAINING
    trained_model = train_worker(0, 1, vocab, config)

    # OUTPUT
    print(trained_model)
