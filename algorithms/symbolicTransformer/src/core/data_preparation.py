import os
from os.path import exists

import spacy
import torch
from torchtext.vocab import build_vocab_from_iterator
from common.constant import Tag, pad_idx

from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from
from algorithms.symbolicTransformer.src.tools.helper import tokenize


def retrieve_phoenix_dataset(env, application_path):
    """
    Extract the parallel sentences and glosses from database
    :param env: a chosen environment {train, test, dev}
    :param application_path: the given code path
    :return: corpus dataframe
    """
    db_dataset = []
    for d in retrieve_mysql_datas_from(env.value[1], application_path):

        src_line = d.get("src")
        trg_line = d.get("tgt")

        if src_line != 'text_content' and trg_line != 'glosses_content':
            db_dataset.append((src_line, trg_line))

    return db_dataset


def load_tokenizers():
    """
    get a French doc (https://spacy.io/api/doc) object from internet if not already present.
    :return: a spacy object
    """
    try:
        spacy_fr = spacy.load("fr_core_news_sm")
    except IOError:
        os.system("python -m spacy download fr_core_news_sm")
        spacy_fr = spacy.load("fr_core_news_sm")

    return spacy_fr


class Vocab:
    """
    Create source and target torchtext vocabulary (named vocab_file_name) which are
    yield into tokens filled (by dataset text or glosses) into itos units by
    build_vocab_from_iterator (https://pytorch.org/text/stable/vocab.html)
    """
    def __init__(self, token_fr, config, env):
        self.src = None
        self.tgt = None
        self.environment = env
        self.tokens = token_fr
        self.vocab_handler(
            config["model_path"]+config["vocab_file_name"],
            config["application_path"])

    def vocab_handler(self, file_path, application_path):
        if not exists(file_path):
            self.src, self.tgt = self.vocab_builder(application_path)
            self.save_vocab(file_path)
        else:
            self.src, self.tgt = torch.load(file_path)

    def vocab_builder(self, application_path):

        learning_corpus = retrieve_phoenix_dataset(self.environment, application_path)
        special_tag = [str(Tag.START.value), str(Tag.STOP.value), str(Tag.BLANK.value), str(Tag.UNKNOWN.value)]

        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        print("Building text Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(learning_corpus, self.tokenize_fr, index=0),
            min_freq=2,
            specials=special_tag,
        )

        print("Building glosses Vocabulary ...")
        vocab_tgt = build_vocab_from_iterator(
            yield_tokens(learning_corpus, self.tokenize_fr, index=1),
            min_freq=1,
            specials=special_tag,
        )

        vocab_src.set_default_index(vocab_src[str(Tag.UNKNOWN.value)])
        vocab_tgt.set_default_index(vocab_tgt[str(Tag.UNKNOWN.value)])

        return vocab_src, vocab_tgt

    def tokenize_fr(self, text):
        return tokenize(text, self.tokens)

    def untokenize_src(self, text):
        return [self.src.get_itos()[x] for x in text if x != pad_idx]

    def untokenize_tgt(self, text):
        return [self.tgt.get_itos()[x] for x in text if x != pad_idx]

    def save_vocab(self, file_path):
        if file_path is not None and self.src is not None and self.tgt is not None:
            torch.save((self.src, self.tgt), file_path)

    @staticmethod
    def pretty_print_token(txt, tokens):
        print(
            txt
            + " ".join(tokens).replace("\n", "")
        )
