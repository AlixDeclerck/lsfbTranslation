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

phrases = [
    "La perfection réside dans l'amour que l'on porte.",
    "Chaque nouvelle journée est une instance unique comme les autres.",
    "La définition du mot femme est aussi étendue que le coeur des hommes sait s'ouvrir.",
    "Je considère que Edward Snowden est un héros et le fait qu'il n'ai trouvé asile qu'en Russie me terrifie!",
]

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

def main():
    # learning_corpus = retrieve_phoenix_dataset(EnvType.DEV, application_path)
    learning_corpus = phrases
    shallow_parsing_verb(learning_corpus[0])


global nlp, application_path

dir_separator = "/"                     # linux folder structure
nlp = spacy.load("fr_core_news_sm")     # nlp : the doc object
args = docopt(__doc__)                  # read application parameters

# Construct application path
application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

if __name__ == '__main__':
    main()
