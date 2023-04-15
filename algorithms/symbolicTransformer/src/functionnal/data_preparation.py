import os
from os.path import exists
import copy
import spacy
import torch
import torchtext
from torchtext.vocab import build_vocab_from_iterator, vocab
from common.constant import Tag, Corpus, EnvType

from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from


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


def load_spacy_tokenizers():
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
    def __init__(self, tokens, config):
        self.src = None
        self.src_vector = None
        self.tgt = None
        self.tgt_vector = None
        self.token_fr = tokens[0]
        self.token_en = tokens[1]
        self.archi_dev_mode = config["architecture_dev_mode"]
        self.vocab_handler(
            config["model_path"]+config["vocab_file_name"],
            config["application_path"],
            config["dimension"],
            bool(config["fast_text_corpus"]))

    def vocab_handler(self, file_path, application_path, dim, is_fast_text):
        """handle the vocabulary (create or load if exists) """
        if not exists(file_path):
            self.src, self.tgt = self.vocab_builder_parallels(application_path)
            self.save_vocab(file_path)
        else:
            self.src, self.tgt = torch.load(file_path)

        if is_fast_text:
            self.src_vector = self.create_parallels_embeddings(dim)

    def create_parallels_embeddings(self, dim):
        # Corpus initialization
        fast_text_corpus = torchtext.vocab.FastText(language='fr')
        corpus_dict = fast_text_corpus.stoi
        corpus_embeddings = fast_text_corpus.vectors

        res = torch.cat((
            torch.zeros(corpus_embeddings.shape[1], 1),
            torch.unsqueeze(corpus_embeddings[0], 1),
            torch.zeros(corpus_embeddings.shape[1], 1),
            torch.zeros(corpus_embeddings.shape[1], 1)), 1
        )

        for word in self.src.vocab.itos_[4:]:
            if word in corpus_dict:
                index = corpus_dict[word]
                res = torch.cat((res, torch.unsqueeze(corpus_embeddings[index], 1)), 1)
            else:
                res = torch.cat((res, torch.zeros(corpus_embeddings.shape[1], 1)), 1)

        output = torch.nn.Linear(corpus_embeddings.shape[1], dim, bias=False, device=None, dtype=None)
        return output(res.T)

    @staticmethod
    def vocab_builder_fast_text(dim):

        # Corpus initialization
        fast_text_corpus = torchtext.vocab.FastText(language='fr')
        corpus_dict = fast_text_corpus.stoi
        corpus_embeddings = torch.matmul(fast_text_corpus.vectors, torch.ones(fast_text_corpus.vectors.shape[1], dim))

        special_tags = {
            str(Tag.START.value[0]): Tag.START.value[1],
            str(Tag.STOP.value[0]): Tag.STOP.value[1],
            str(Tag.BLANK.value[0]): Tag.BLANK.value[1],
            str(Tag.UNKNOWN.value[0]): Tag.UNKNOWN.value[1]
        }

        special_embeddings = torch.cat((
            torch.zeros(dim, 1),
            torch.unsqueeze(corpus_embeddings[0], 1),
            torch.zeros(dim, 1),
            torch.zeros(dim, 1)), 1
        )

        # taking information (value, embedding, index)
        print("Building FRENCH Vocabulary from fast text")
        coma_tag = (list(corpus_dict.keys())[1], copy.deepcopy(corpus_embeddings[1]), corpus_dict.pop("//www"))
        de_tag = (list(corpus_dict.keys())[2], copy.deepcopy(corpus_embeddings[2]), corpus_dict.pop("#"))
        double_quotes_tag = (list(corpus_dict.keys())[3], copy.deepcopy(corpus_embeddings[3]), corpus_dict.pop("www"))

        # first values permutations
        corpus_dict[coma_tag[0]] = coma_tag[2]
        corpus_embeddings[coma_tag[2]] = coma_tag[1]
        corpus_dict[de_tag[0]] = de_tag[2]
        corpus_embeddings[de_tag[2]] = de_tag[1]
        corpus_dict[double_quotes_tag[0]] = double_quotes_tag[2]
        corpus_embeddings[double_quotes_tag[2]] = double_quotes_tag[1]

        # remove first entry which is into special tags
        corpus_dict.pop(Tag.STOP.value[0])
        corpus_embeddings = corpus_embeddings[1:]

        # add special character @ first place
        corpus_dict.update(special_tags)

        corpus_embeddings = torch.cat((
            special_embeddings.T,
            corpus_embeddings)
        )

        return corpus_dict, corpus_embeddings

    def vocab_builder_parallels(self, application_path):
        """create a vocabulary (mapping between token and index - one hot vector)"""
        special_tag = [str(Tag.START.value[0]), str(Tag.STOP.value[0]), str(Tag.BLANK.value[0]), str(Tag.UNKNOWN.value[0])]
        learning_corpus = []
        for env in EnvType:
            learning_corpus += retrieve_conte_dataset(env.value, application_path)

        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        print("Building FRENCH Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(learning_corpus, self.tokenize_fr, index=Corpus.TEXT_FR.value[1]),
            min_freq=1,
            specials=special_tag,
        )

        if self.archi_dev_mode:
            print("Building ENGLISH Vocabulary ...")
            vocab_tgt = build_vocab_from_iterator(
                yield_tokens(learning_corpus, self.tokenize_en, index=Corpus.TEXT_EN.value[1]),
                min_freq=1,
                specials=special_tag)
        else:
            print("Building GLOSS Vocabulary ...")
            vocab_tgt = build_vocab_from_iterator(
                yield_tokens(learning_corpus, self.tokenize_gloss, index=Corpus.GLOSS_LSF.value[1]),
                min_freq=1,
                specials=special_tag)

        vocab_src.set_default_index(vocab_src[str(Tag.UNKNOWN.value[0])])
        vocab_tgt.set_default_index(vocab_tgt[str(Tag.UNKNOWN.value[0])])

        return vocab_src, vocab_tgt

    def vocab_embedder_fast_text_fr(self):
        """update vocabulary and embeddings with a d=300 vectors embedding and its vocabulary """
        # vocab_tgt = torchtext.vocab.FastText(language='en')
        vocab_src = torchtext.vocab.FastText(language='fr')
        special_tag = [str(Tag.START.value[0]), str(Tag.STOP.value[0]), str(Tag.BLANK.value[0]), str(Tag.UNKNOWN.value[0])]

        self.src.vocab.itos_.clear()
        self.src.vocab.itos_.append(special_tag+vocab_src.itos[1:])
        self.src_vector = torch.cat((torch.cat((torch.zeros(300, 1), torch.unsqueeze(vocab_src.vectors[0], 1), torch.zeros(300, 1), torch.zeros(300, 1)), 1).T, vocab_src.vectors[1:]))

    def tokenize_fr(self, text):
        return self.tokenize(text, self.token_fr)

    def tokenize_gloss(self, text):
        return self.tokenize(text, self.token_fr)

    def tokenize_en(self, text):
        return self.tokenize(text, self.token_en)

    def unembed_src(self, text):
        """Index to string on the source vocabulary"""
        return [self.src.get_itos()[x] for x in text if x != Tag.BLANK.value[1]]

    def unembed_tgt(self, text):
        """Index to string on the target vocabulary"""
        return [self.tgt.get_itos()[x] for x in text if x != Tag.BLANK.value[1]]

    def embed_tgt(self, text):
        """String to index on the target vocabulary"""
        return [self.tgt[x] for x in text]

    def save_vocab(self, file_path):
        """Local persist at provided path"""
        if file_path is not None and self.src is not None and self.tgt is not None:
            torch.save((self.src, self.tgt), file_path)

    @staticmethod
    def pretty_print_token(txt, tokens):
        print(
            txt
            + " ".join(tokens).replace("\n", "")
        )

    @staticmethod
    def tokenize(text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]
