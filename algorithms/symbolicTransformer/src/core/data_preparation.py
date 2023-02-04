import os
from os.path import exists

import spacy
import torch
from torchtext.vocab import build_vocab_from_iterator

from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from
from algorithms.symbolicTransformer.src.tools.helper import tokenize


def retrieve_phoenix_dataset(env, application_path):
    examples = []

    # Extract the parallel src, trg and file(s) from database
    for d in retrieve_mysql_datas_from(env.value[1], application_path):

        src_line = d.get("src")
        trg_line = d.get("tgt")

        # Create a dataset examples out of the Source, Target Frames and FilesPath
        if src_line != 'text_content' and trg_line != 'glosses_content':
            examples.append((src_line, trg_line))

    return examples


def load_tokenizers():

    try:
        spacy_fr = spacy.load("fr_core_news_sm")
    except IOError:
        os.system("python -m spacy download fr_core_news_sm")
        spacy_fr = spacy.load("fr_core_news_sm")

    return spacy_fr


class Vocab:

    def __init__(self, token_fr, config, env):
        self.vocab_src = None
        self.vocab_tgt = None
        self.environment = env
        self.french_tokens = token_fr
        self.vocab_handler(
            config["model_path"]+config["vocab_file_name"],
            config["application_path"])

    def vocab_handler(self, file_path, application_path):
        if not exists(file_path):
            self.vocab_src, self.vocab_tgt = self.vocab_builder(application_path)
            self.save_vocab(file_path)
        else:
            self.vocab_src, self.vocab_tgt = torch.load(file_path)

    def vocab_builder(self, application_path):

        learning_corpus = retrieve_phoenix_dataset(self.environment, application_path)

        def tokenize_fr(text):
            return tokenize(text, self.french_tokens)

        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        print("Building text Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(learning_corpus, tokenize_fr, index=0),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        print("Building glosses Vocabulary ...")
        vocab_tgt = build_vocab_from_iterator(
            yield_tokens(learning_corpus, tokenize_fr, index=1),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        vocab_src.set_default_index(vocab_src["<unk>"])
        vocab_tgt.set_default_index(vocab_tgt["<unk>"])

        return vocab_src, vocab_tgt

    def save_vocab(self, file_path):
        if file_path is not None and self.vocab_src is not None and self.vocab_tgt is not None:
            torch.save((self.vocab_src, self.vocab_tgt), file_path)
