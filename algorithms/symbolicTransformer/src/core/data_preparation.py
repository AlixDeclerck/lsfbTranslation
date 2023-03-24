import os
from os.path import exists

import spacy
import torch
from torchtext.vocab import build_vocab_from_iterator
from common.constant import Tag, Corpus, TargetMode

from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from
from algorithms.symbolicTransformer.src.tools.helper import tokenize


def retrieve_conte_dataset(selected_environment, application_path):
    """
    Extract the parallel sentences and glosses from database
    :param selected_environment: a chosen string environment {train, test, dev}
    :param application_path: the given code path
    :return: corpus dataframe
    """
    db_dataset = []
    for d in retrieve_mysql_datas_from(selected_environment, application_path):
        db_dataset.append([
            d.get(Corpus.TEXT_FR.value[0]),
            d.get(Corpus.TEXT_EN.value[0]),
            d.get(Corpus.GLOSS_LSF.value[0])
        ])

    return db_dataset


def load_tokenizers():
    """
    get a spacy "doc object" (https://spacy.io/api/doc) from internet if not already present.
    :return: a tuple with French and English spacy object
    """
    try:
        spacy_fr = spacy.load("fr_core_news_sm")
    except IOError:
        os.system("python -m spacy download fr_core_news_sm")
        spacy_fr = spacy.load("fr_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_fr, spacy_en


class Vocab:
    """
    Create source and target torchtext vocabulary (named vocab_file_name) which are
    yield into tokens filled (by dataset text or glosses) into itos units by
    build_vocab_from_iterator (https://pytorch.org/text/stable/vocab.html)
    """
    def __init__(self, tokens, config, env):
        self.src = None
        self.tgt = None
        self.environment = env.value[0]
        self.token_fr = tokens[0]
        self.token_en = tokens[1]
        self.target_mode = config["target_mode"]
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

        learning_corpus = retrieve_conte_dataset(self.environment, application_path)
        special_tag = [str(Tag.START.value[0]), str(Tag.STOP.value[0]), str(Tag.BLANK.value[0]), str(Tag.UNKNOWN.value[0])]

        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        print("Building french Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(learning_corpus, self.tokenize_fr, index=Corpus.TEXT_FR.value[1]),
            min_freq=1,
            specials=special_tag,
        )

        if TargetMode.EN.value == self.target_mode:
            print("Building english Vocabulary ...")
            vocab_tgt = build_vocab_from_iterator(
                yield_tokens(learning_corpus, self.tokenize_en, index=Corpus.TEXT_EN.value[1]),
                min_freq=1,
                specials=special_tag)
        else:
            print("Building gloss Vocabulary ...")
            vocab_tgt = build_vocab_from_iterator(
                yield_tokens(learning_corpus, self.tokenize_gloss, index=Corpus.GLOSS_LSF.value[1]),
                min_freq=1,
                specials=special_tag)

        vocab_src.set_default_index(vocab_src[str(Tag.UNKNOWN.value[0])])
        vocab_tgt.set_default_index(vocab_tgt[str(Tag.UNKNOWN.value[0])])

        return vocab_src, vocab_tgt

    def tokenize_fr(self, text):
        return tokenize(text, self.token_fr)

    def tokenize_gloss(self, text):
        return tokenize(text, self.token_fr)

    def tokenize_en(self, text):
        return tokenize(text, self.token_en)

    def untokenize_src(self, text):
        return [self.src.get_itos()[x] for x in text if x != Tag.BLANK.value[1]]

    def untokenize_tgt(self, text):
        return [self.tgt.get_itos()[x] for x in text if x != Tag.BLANK.value[1]]

    def save_vocab(self, file_path):
        if file_path is not None and self.src is not None and self.tgt is not None:
            torch.save((self.src, self.tgt), file_path)

    @staticmethod
    def pretty_print_token(txt, tokens):
        print(
            txt
            + " ".join(tokens).replace("\n", "")
        )
