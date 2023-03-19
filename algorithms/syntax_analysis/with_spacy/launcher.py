#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    launcher.py parsing --app-path=<file>
"""

import os
import spacy
import textacy
from docopt import docopt
import unicodedata
from algorithms.data_loader.src.dal import EnvType
from algorithms.symbolicTransformer.src.core.data_preparation import retrieve_mysql_datas_from

txt_samples = [
    "La perfection réside dans l'amour que l'on porte.",
    "Chaque nouvelle journée est (une instance) unique comme toutes les autres.",
    "Au 21° siècle, la définition du mot femme s'étend comme le coeur des hommes s'ouvre.",
    "Je considère Edward Snowden comme un héros et il n'a trouvé asile qu'en Russie. Cela me terrifie!",
]

class SubPhrase:
    def __init__(self):
        self.event = []
        self.classificator = []
        self.subject = []
        self.action = []
        self.tense = []
        self.unclassified = []
        self.items_number = 0

    def add_word(self, token):
        if token.pos_ in ["AUX"]:
            self.tense.append(token.morph.get("Tense")[0])

        if token.pos_ in ["ADV", "NUM"]:  # EVENT ("ADP", ?)
            if "Neg" in token.morph.get("Polarity"):
                self.event.append("PAS")
            else:
                self.event.append(token)

        if token.pos_ in ["PART"]:  # CLASSIFICATOR
            self.classificator.append(token)

        if token.pos_ in ["NOUN", "PROPN", "ADJ"]:  # SUBJECT (""PRON"
            self.subject.append(token)

        if token.pos_ in ["VERB"]:  # ACTION
            self.action.append(token.lemma_)
            self.tense.append(token.morph.get("Tense")[0])

        else:  # UNCLASSIFIED
            self.unclassified.append(token)

        self.items_number += 1

    def __len__(self):
        return self.items_number


class SpacyPhrase:
    """
    Provide some functionalities to walk annotated glosses list
    """
    def __init__(self, txt):
        self.raw_txt = txt          # a phrase with nlp spacy format
        self.filtered = []          # filtered words during preprocessing
        self.phrases = []           # phrase after pre_preprocessing
        self.scene = []             # a list of sub_phrases
        self.tokens = []            # a list of tokens to display

    """
    constructing glosses from raw_text 
    making a preprocessing into glosses dictionary:
    - filtering {DETERMINANT, SYMBOL} and stop words
    """
    def preprocessing(self):
        for token in self.raw_txt:
            if token.text in ["°"]:
                self.filtered.append(token)  # stop words
            elif token.pos_ in ["PUNCT"] and token.text not in [".", ";", ",", ":"]:
                self.filtered.append(token)
            elif token.pos_ in ["DET", "SYM"]:
                self.filtered.append(token)
            else:
                self.phrases.append(token)

    """
    do a scenes segmentation by punctuation signs into sub phrases
    """
    def scene_construction(self):
        sub_phrase = SubPhrase()
        for token in self.phrases:
            if "PUNCT" == token.pos_ or "et" == token.text:
                if sub_phrase.__len__() > 0:
                    self.scene.append(sub_phrase)
                    sub_phrase = SubPhrase()
            else:
                sub_phrase.add_word(token)
        if sub_phrase.__len__() > 0:
            self.scene.append(sub_phrase)

    """
    using basic grammatical scene construction steps 
    """
    def make_sentence(self):
        self.tokens = []
        for sub_phrase in self.scene:
            for s in sub_phrase.event:
                self.tokens.append(s)
            for s in sub_phrase.classificator:
                self.tokens.append(s)
            for s in sub_phrase.subject:
                self.tokens.append(s)
            for s in sub_phrase.action:
                self.tokens.append(s)

    """
    create a sentence from tokens and print it
    """
    def write(self):
        # print(f"1 phrase with {self.__len__()} sub phrases")
        res = ""
        for t in self.tokens:
            if not isinstance(t, str):
                res += t.text+" "
            else:
                res += t+" "

        print(self.raw_txt)
        print(u"".join([x for x in unicodedata.normalize("NFKD", res).upper() if not unicodedata.combining(x)]))
        print("-----")

    def __len__(self):
        return len(self.scene)


# --------------------------------------------------------------------------

args = docopt(__doc__)                  # read application parameters
dir_separator = "/"                     # linux folder structure
nlp = spacy.load("fr_core_news_sm")     # nlp : the doc object

# Construct application path
application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

def main():
    # learning_corpus = retrieve_mysql_datas_from(EnvType.DEV, application_path)

    for txt in txt_samples:
        phrases = SpacyPhrase(nlp(txt))
        phrases.preprocessing()
        phrases.scene_construction()
        phrases.make_sentence()
        phrases.write()


if __name__ == '__main__':
    main()
