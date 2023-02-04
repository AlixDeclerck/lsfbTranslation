#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    bpe.py train --app-path=<file>
"""


# sources & infos
# https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0
# https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10

import re
from collections import Counter, defaultdict

from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from docopt import docopt
import os
from algorithms.data_loader.src.dal import EnvType
from common.constant import dir_separator

from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers
from algorithms.symbolicTransformer.src.tools.helper import load_config


def build_vocab(corpus: str) -> dict:
    """Step 1. Build vocab from text corpus"""

    # Separate each char in word by space and add mark end of token
    tokens = [" ".join(word) + " </w>" for word in corpus.split()]

    # Count frequency of tokens in corpus
    vocab = Counter(tokens)

    return vocab


def get_stats(vocab: dict) -> dict:
    """Step 2. Get counts of pairs of consecutive symbols"""

    pairs = defaultdict(int)
    for word, frequency in vocab.items():
        symbols = word.split()

        # Counting up occurrences of pairs
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency

    return pairs


def merge_vocab(pair: tuple, v_in: dict) -> dict:
    """Step 3. Merge all occurrences of the most frequent pair"""

    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in v_in:
        # replace most frequent pair in all vocabulary
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out


# ---------------------------------------------------------------

if __name__ == '__main__':

    args = docopt(__doc__)
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator
    learning_configuration = load_config()

    token_fr = load_tokenizers()
    vocab = Vocab(load_tokenizers(), learning_configuration, application_path, EnvType.DEV)

    vocab = build_vocab(vocab.vocab_src)  # Step 1

    num_merges = 50  # Hyperparameter
    for i in range(num_merges):

        pairs = get_stats(vocab)  # Step 2

        if not pairs:
            break

        # step 3
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
