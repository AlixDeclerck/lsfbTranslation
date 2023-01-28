import os
from os.path import exists

import numpy as np
import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torchdata.datapipes.iter import Mapper
from torchtext.vocab import build_vocab_from_iterator
from algorithms.data_loader.src.dal import EnvType

from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from
from algorithms.symbolicTransformer.src.tools.helper import tokenize


def retrieve_phoenix_dataset(env):
    examples = []

    # Extract the parallel src, trg and file(s) from database
    for d in retrieve_mysql_datas_from(env):

        src_line = d.get("src")
        trg_line = d.get("tgt")

        # Create a dataset examples out of the Source, Target Frames and FilesPath
        if src_line != 'text_content' and trg_line != 'glosses_content':
            examples.append((src_line, trg_line))

    return examples


# def retrieve_mtnt_dataset():
#     ds = tfds.load('mtnt', split='train', shuffle_files=True)
#     assert isinstance(ds, tf.data.Dataset)
#     print(ds)
#
#     builder = tfds.builder('mtnt')
#     builder.download_and_prepare()
#     ds = builder.as_dataset(split='train', shuffle_files=True)
#
#     train = ds
#     map_train = Mapper(train, lambda x: x)
#     val = []
#     for i, j in enumerate(list(map_train)):
#         str_fr = str(np.array(j.get("dst")))[2:-1]
#         str_en = str(np.array(j.get("src")))[2:-1]
#         res = str_fr, str_en
#         val.append(res)
#     return val


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def load_tokenizers():

    # for retrieve_phoenix_dataset
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    # uncomment for french
    # try:
    #     spacy_fr = spacy.load("fr_core_news_sm")
    # except IOError:
    #     os.system("python -m spacy download fr_core_news_sm")
    #     spacy_fr = spacy.load("fr_core_news_sm")

    return spacy_de


def build_vocabulary(spacy_de):

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    # def tokenize_fr(text):
    #     return tokenize(text, spacy_fr)

    learning_corpus = retrieve_phoenix_dataset(EnvType.TRAIN.value[1])

    print("Building text Vocabulary ...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(learning_corpus, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building glosses Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(learning_corpus, tokenize_de, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, config):
    file_path = config["model_path"]+config["vocab_file_name"]

    if not exists(file_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de)
        torch.save((vocab_src, vocab_tgt), file_path)
    else:
        vocab_src, vocab_tgt = torch.load(file_path)

    print("Loading finished!")
    print("source vocabulary size : ", len(vocab_src))
    print("target vocabulary size : ", len(vocab_tgt))
    return vocab_src, vocab_tgt
