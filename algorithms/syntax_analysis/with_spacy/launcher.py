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

def shallow_parsing_verb(txt):
    patterns = [{"POS": "AUX"}, {"POS": "VERB"}]
    about_talk_doc = textacy.make_spacy_doc(
        txt, lang="en_core_web_sm"
    )
    verb_phrases = textacy.extract.token_matches(
        about_talk_doc, patterns=patterns
    )

    print("Print all verb phrases")
    for chunk in verb_phrases:
        print(chunk.text)

    print("\nExtract noun phrase to explain what nouns are involved")
    for chunk in about_talk_doc.noun_chunks:
        print(chunk)

def part_of_speech(txt):
    for token in txt:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)


txt_samples = [
    "La perfection réside dans l'amour que l'on porte.",
    "Chaque nouvelle journée est (une instance) unique comme toutes les autres.",
    "La définition du mot femme est aussi étendue que le coeur des hommes sait s'ouvrir.",
    "Je considère que Edward Snowden est un héros et le fait qu'il n'ai trouvé asile qu'en Russie me terrifie!",
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

        if token.pos_ in ["ADJ", "ADP", "ADV", "NUM"]:  # EVENT
            self.event.append(token)

        if token.pos_ in ["PART"]:  # CLASSIFICATOR
            self.classificator.append(token)

        if token.pos_ in ["NOUN", "PRON", "PROPN"]:  # SUBJECT
            self.subject.append(token)

        if token.pos_ in ["VERB"]:  # ACTION
            self.action.append(token)

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
    - filtering {DETERMINANT, SYMBOL}
    """
    def preprocessing(self):
        for token in self.raw_txt:
            if token.pos_ in ["PUNCT"] and token.text not in [".", ";", ",", ":"]:
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
            if ["PUNCT"] == token.pos_:  # todo: ajouter le "and"
                if sub_phrase.__len__() > 0:
                    self.scene.append(sub_phrase)
                    sub_phrase = SubPhrase()
            else:
                sub_phrase.add_word(token)
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
        res = "->"
        for t in self.tokens:
            res += " "+t.text

        print(self.raw_txt)
        print(f"".join([x for x in unicodedata.normalize("NFKD", res).upper() if not unicodedata.combining(x)]))
        print("-----")


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
