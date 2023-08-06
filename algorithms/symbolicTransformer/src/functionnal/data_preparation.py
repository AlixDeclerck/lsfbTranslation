import os
from os.path import exists

import spacy
import torch
import torchtext
import unidecode
from torchtext.vocab import build_vocab_from_iterator

from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from
from common.constant import Tag, Corpus, EnvType, Dialect

"""
This code was genuinely inspired by 
"the progressive transformer" (Ben Saunders et al.) and "the annotated transformer" (Huang / Rush et al.) 
and was rewritten from scratch by Alix Declerck et al. UMONS
We create a vocabulary to be used by the symbolicTransformer
"""

def retrieve_conte_dataset(selected_environment, application_path, selected_db, dialect, english_output=False, multi_source=False, limit=10000):
    """
    Extract the parallel sentences and glosses from database
    :param selected_environment: a chosen string environment {train, test, dev}
    :param application_path: the given code path
    :param selected_db: db_dev or db_test
    :param dialect: to choose which glosses are taken {0:"both", 1:"LSF", 2:"generated"}
    :param english_output: False mean we use glosses
    :param multi_source : If true we take booth generated and FR source when possible
    :param limit : To limit the request output
    :return: corpus dataframe
    """

    print("retrieve conte dataset for : ", dialect.value[1])

    db_dataset = []
    for d in retrieve_mysql_datas_from(selected_environment, application_path, selected_db, dialect_selection=dialect.value[0], src_multi=multi_source, request_limit=limit):
        data = [d.get(Corpus.TEXT_FR.value[0]), d.get(Corpus.TEXT_EN.value[0]), d.get(Corpus.GLOSS_LSF.value[0])]
        if english_output or data[2] != "":
            db_dataset.append(data)

    return db_dataset

def retrieve_corpus_txt(path):
    with open(path) as f:
        res = f.readlines()
    return res

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
    def __init__(self, config):
        tokens = load_spacy_tokenizers()
        self.src = None
        self.src_vector = None
        self.tgt = None
        self.tgt_vector = None
        self.french_tokenizer = tokens[0]
        self.english_tokenizer = tokens[1]
        self.vocab_txt = config["learning_config"]["vocab_txt"]
        self.is_english_output = config["learning_config"]["english_output"]
        self.multi_source = config["learning_config"]["multi_sources"]
        self.row_limit = config["learning_config"]["row_limit"]
        for dia in Dialect:
            if config["learning_config"]["dialect_selection"] == dia.value[0]:
                self.dialect_selection = dia
                break

        for dia in Dialect:
            if config["learning_config"]["vocab_dialect"] == dia.value[0]:
                self.vocab_dialect = dia
                break

        self.vocab_name = config["configuration_path"]["model_path"]+config["configuration_path"]["vocab_file_name"]
        self.application_path = config["configuration_path"]["application_path"]
        self.dimension = config["hyper_parameters"]["dimension"]
        self.is_fasttext = bool(config["learning_config"]["fast_text_corpus"])
        self.is_english_output = bool(config["learning_config"]["english_output"])
        self.selected_db = str(config["configuration_path"]["selected_db"])
        self.txt_corpus = str(config["configuration_path"]["txt_corpus"])
        self.join_vocab = bool(config["learning_config"]["join_vocab"])

    def create(self):
        self.vocab_handler(self.vocab_name, self.application_path, self.selected_db)

    def retrieve_from_disk(self):
        if exists(self.vocab_name):
            self.src, self.tgt = torch.load(self.vocab_name)
            if self.is_fasttext:
                self.src_vector = self.create_src_embeddings(self.dimension)
                self.tgt_vector = self.create_tgt_embeddings(self.dimension, self.is_english_output)
        else:
            return "No vocabulary in ", self.vocab_name

    def vocab_handler(self, file_path, application_path, selected_db):
        """handle the vocabulary creation """
        if not exists(file_path):
            self.src, self.tgt = self.vocab_builder_parallels(application_path, selected_db)
            self.save_vocab(file_path)

        else:
            return "A Vocab exist in ", file_path

    def vocab_builder_parallels(self, application_path, selected_db):
        """
            create a vocabulary (mapping between token and index - one hot vector)
            We construct the vocabulary with valid glosses
        """
        special_tag = [str(Tag.START.value[0]), str(Tag.STOP.value[0]), str(Tag.BLANK.value[0]), str(Tag.UNKNOWN.value[0])]
        learning_corpus = []
        for env in EnvType:
            learning_corpus += retrieve_conte_dataset(env.value, application_path, selected_db, self.vocab_dialect, self.is_english_output, self.multi_source, self.row_limit)

        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        print("Building FRENCH Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(learning_corpus, self.tokenize_fr, index=Corpus.TEXT_FR.value[1]),
            min_freq=1,
            specials=special_tag,
        )

        if self.is_english_output:
            print("Building ENGLISH Vocabulary ...")
            vocab_tgt = build_vocab_from_iterator(
                yield_tokens(learning_corpus, self.tokenize_en, index=Corpus.TEXT_EN.value[1]),
                min_freq=1,
                specials=special_tag)
        else:
            print("Building GLOSS Vocabulary ...")

            if self.vocab_txt:
                lsfb_corpus = retrieve_corpus_txt(application_path+self.txt_corpus)
                learning_corpus.append(["", "", lsfb_corpus[0]])

            vocab_tgt = build_vocab_from_iterator(
                yield_tokens(learning_corpus, self.tokenize_gloss, index=Corpus.GLOSS_LSF.value[1]),
                min_freq=1,
                specials=special_tag)

        vocab_src.set_default_index(vocab_src[str(Tag.UNKNOWN.value[0])])
        vocab_tgt.set_default_index(vocab_tgt[str(Tag.UNKNOWN.value[0])])

        return vocab_src, vocab_tgt

    def create_src_embeddings(self, dim):
        """
            retrieve fasttext french embeddings for the source property
            :param dim: the embeddings goes (token, 300)x(300, dim) with a Linear
            :return: source embeddings
        """
        # corpus initialization
        fast_text_corpus = torchtext.vocab.FastText(language='fr')

        # embedding
        embedding, dimension = self.align_tokens(fast_text_corpus.stoi, fast_text_corpus.vectors, self.src.vocab.itos_[4:])
        output = torch.nn.Linear(dimension, dim, bias=False, device=None, dtype=None)

        return output(embedding.T)

    def create_tgt_embeddings(self, dim, is_en):
        """
            retrieve fasttext english / gloss embeddings for the source property
            :param dim: the embeddings goes (token, 300)x(300, dim) with a Linear
            :param is_en: target configuration : english or glosses
            :return: target embeddings
        """
        if is_en:
            fast_text_corpus = torchtext.vocab.FastText(language='en')
            corpus_stoi = fast_text_corpus.stoi
            corpus_vectors = fast_text_corpus.vectors
        else:
            fast_text_corpus = torchtext.vocab.FastText(language='fr')
            corpus_stoi, corpus_vectors = self.add_embeddings(fast_text_corpus)

        embedding, dimension = self.align_tokens(corpus_stoi, corpus_vectors, self.tgt.vocab.itos_[4:])
        output = torch.nn.Linear(dimension, dim, bias=False, device=None, dtype=None)

        return output(embedding.T)

    def tokenize_fr(self, text):
        return self.tokenize(text, self.french_tokenizer)

    def tokenize_gloss(self, text):
        return self.tokenize(text, self.french_tokenizer)

    def tokenize_en(self, text):
        return self.tokenize(text, self.english_tokenizer)

    def unembed_src(self, text):
        """Index to string on the source vocabulary"""
        return [self.src.get_itos()[x] for x in text if x != Tag.BLANK.value[1]]

    def unembed_tgt(self, text):
        """Index to string on the target vocabulary"""
        return [self.tgt.get_itos()[x] for x in text if x != Tag.BLANK.value[1]]

    def stoi_tgt(self, text):
        """String to index on the target vocabulary"""
        return [self.tgt[x] for x in text]

    def save_vocab(self, file_path):
        """Local persist at provided path"""
        if file_path is not None and self.src is not None and self.tgt is not None:
            torch.save((self.src, self.tgt), file_path)

    @staticmethod
    def add_embeddings(original_corpus):
        """
        We use the fasttext french embedding to create glosses embeddings
        Side effect : when a word (key) exist with different accents we keep only the last embedding (value)
        Then we recreate a new stoi / vectors
        :param original_corpus:
        :return: stoi, vector
        """
        new_stoi = {}
        corp = list(original_corpus.stoi.items())[1:]
        corp.reverse()

        for x in corp:
            new_key = unidecode.unidecode(x[0]).upper()
            new_embedding = original_corpus.vectors[x[1]]
            new_stoi.update({new_key: new_embedding})

        res_stoi = {'</s>': 0}
        res_vectors = [original_corpus.vectors[0]]
        items = list(new_stoi.items())[1:]
        items.reverse()
        for index, x in enumerate(items):
            res_vectors.append(x[1])
            res_stoi.update({x[0]: index})

        return res_stoi, torch.stack(res_vectors)

    @staticmethod
    def align_tokens(corpus_dict, corpus_embeddings, words):
        res = torch.cat((
            torch.zeros(corpus_embeddings.shape[1], 1),
            torch.unsqueeze(corpus_embeddings[0], 1),
            torch.zeros(corpus_embeddings.shape[1], 1),
            torch.zeros(corpus_embeddings.shape[1], 1)), 1
        )

        for word in words:
            if word in corpus_dict:
                index = corpus_dict[word]
                res = torch.cat((res, torch.unsqueeze(corpus_embeddings[index], 1)), 1)
            else:
                res = torch.cat((res, torch.zeros(corpus_embeddings.shape[1], 1)), 1)

        return res, corpus_embeddings.shape[1]

    @staticmethod
    def pretty_print_token(txt, tokens):
        print(
            txt
            + " ".join(tokens).replace("\n", "")
        )

    @staticmethod
    def tokenize(text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]
