import os
from os.path import exists
import copy
import spacy
import torch
import torchtext
import unidecode
from torchtext.vocab import build_vocab_from_iterator
from common.constant import Tag, Corpus, EnvType, Dialect
from algorithms.data_loader.src.retrieve_data import retrieve_mysql_datas_from

"""
This code was genuinely inspired by 
"the progressive transformer" (Ben Saunders et al.) and "the annotated transformer" (Huang / Rush et al.) 
and was rewritten from scratch by Alix Declerck et al. UMONS
We create a vocabulary to be used by the symbolicTransformer
"""

def retrieve_conte_dataset(selected_environment, application_path, selected_db, dialect=0):
    """
    Extract the parallel sentences and glosses from database
    :param selected_environment: a chosen string environment {train, test, dev}
    :param application_path: the given code path
    :param selected_db: db_dev or db_test
    :param dialect: to choose which glosses are taken {0:"both", 1:"LSF", 2:"generated"}
    :return: corpus dataframe
    """
    db_dataset = []
    for d in retrieve_mysql_datas_from(selected_environment, application_path, selected_db, dialect_selection=dialect):
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
    def __init__(self, config):
        tokens = load_spacy_tokenizers()
        self.src = None
        self.src_vector = None
        self.tgt = None
        self.tgt_vector = None
        self.french_tokenizer = tokens[0]
        self.english_tokenizer = tokens[1]
        self.english_output = config["learning_config"]["english_output"]
        self.dialect_selection = config["learning_config"]["dialect_selection"]
        self.vocab_handler(
            config["configuration_path"]["model_path"]+config["configuration_path"]["vocab_file_name"],
            config["configuration_path"]["application_path"],
            config["hyper_parameters"]["dimension"],
            bool(config["learning_config"]["fast_text_corpus"]),
            bool(config["learning_config"]["english_output"]),
            str(config["configuration_path"]["selected_db"])
        )

    def vocab_handler(self, file_path, application_path, dim, is_fast_text, is_en, selected_db):
        """handle the vocabulary (create or load if exists) """
        if not exists(file_path):
            self.src, self.tgt = self.vocab_builder_parallels(application_path, selected_db)
            self.save_vocab(file_path)
        else:
            self.src, self.tgt = torch.load(file_path)

        if is_fast_text:
            self.src_vector = self.create_src_embeddings(dim)
            self.tgt_vector = self.create_tgt_embeddings(dim, is_en)

    def vocab_builder_parallels(self, application_path, selected_db):
        """
            create a vocabulary (mapping between token and index - one hot vector)
            We construct the vocabulary with valid glosses
        """
        special_tag = [str(Tag.START.value[0]), str(Tag.STOP.value[0]), str(Tag.BLANK.value[0]), str(Tag.UNKNOWN.value[0])]
        learning_corpus = []
        for env in EnvType:
            learning_corpus += retrieve_conte_dataset(env.value, application_path, selected_db, Dialect.LSF.value)

        def yield_tokens(data_iter, tokenizer, index):
            for from_to_tuple in data_iter:
                yield tokenizer(from_to_tuple[index])

        print("Building FRENCH Vocabulary ...")
        vocab_src = build_vocab_from_iterator(
            yield_tokens(learning_corpus, self.tokenize_fr, index=Corpus.TEXT_FR.value[1]),
            min_freq=1,
            specials=special_tag,
        )

        if self.english_output:
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

    def create_src_embeddings(self, dim):
        """
            retrieve fasttext french embeddings for the source property
            :param dim: the embeddings goes (token, 300)x(300, dim) with a Linear
            :return: source embeddings
        """
        # corpus initialization
        fast_text_corpus = torchtext.vocab.FastText(language='fr')

        # embedding
        embedding, dimension = self.update_embeddings(fast_text_corpus.stoi, fast_text_corpus.vectors, self.src.vocab.itos_[4:])
        output = torch.nn.Linear(dimension, dim, bias=False, device=None, dtype=None)

        return output(embedding.T)

    def create_tgt_embeddings(self, dim, is_en):
        """
            retrieve fasttext english / gloss embeddings for the source property
            :param dim: the embeddings goes (token, 300)x(300, dim) with a Linear
            :param is_en: target configuration : english or glosses
            :return: target embeddings
        """
        # corpus initialization
        if is_en:
            fast_text_corpus = torchtext.vocab.FastText(language='en')
            corpus_stoi = fast_text_corpus.stoi
            corpus_vectors = fast_text_corpus.vectors
        else:
            fast_text_corpus = torchtext.vocab.FastText(language='fr')
            corpus_stoi, corpus_vectors = self.gloss_the_corpus(fast_text_corpus)

        # embedding
        embedding, dimension = self.update_embeddings(corpus_stoi, corpus_vectors, self.tgt.vocab.itos_[4:])
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
    def gloss_the_corpus(original_corpus):
        """
        We use the fasttext french embedding to create glosses embeddings
        Side effect : when a word (key) exist with different accents we keep only the last embedding (value)
        Then we recreate a new stoi / vectors
        :param original_corpus:
        :return: stoi, vector
        """
        new_stoi = {'</s>': original_corpus.vectors[0]}

        for x in list(original_corpus.stoi.items())[1:]:
            new_key = unidecode.unidecode(x[0]).upper()
            new_embedding = original_corpus.vectors[x[1]]
            new_stoi.update({new_key: new_embedding})

        res_stoi = {'</s>': 0}
        res_vectors = [original_corpus.vectors[0]]
        items = list(new_stoi.items())[1:]
        for index, x in enumerate(items, start=1):
            res_vectors.append(x[1])
            res_stoi.update({x[0]: index})

        return res_stoi, torch.stack(res_vectors)

    @staticmethod
    def update_embeddings(corpus_dict, corpus_embeddings, words):
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
    def vocab_builder_fast_text(dim):
        """A complete vocab builder that generate 10^6 embedded tokens, need big GPU"""
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

    @staticmethod
    def tokenize(text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]
