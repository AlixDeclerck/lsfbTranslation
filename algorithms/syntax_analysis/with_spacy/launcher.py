#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    launcher.py contes --mode=string --app-path=<file>
"""

import os
import spacy
from docopt import docopt
from phrases import SpacyPhrase
from data.conte import samples
from algorithms.data_loader.src.retrieve_data import retrieve_mysql_conte, show_mysql_conte
from common.constant import Corpus

txt_samples = [
    "La perfection réside dans l'amour que l'on porte.",
    "Chaque nouvelle journée est (une instance) unique comme toutes les autres.",
    "Au 21° siècle, la définition du mot femme s'étend comme le coeur des hommes s'ouvre.",
    "Je considère Edward Snowden comme un héros et il n'a trouvé asile qu'en Russie. Cela me terrifie!",
]

# --------------------------------------------------------------------------

args = docopt(__doc__)                  # read application parameters
dir_separator = "/"                     # linux folder structure
nlp = spacy.load("fr_core_news_sm")     # nlp : the doc object

# Construct application path
application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

def format_nbr(num):
    if num < 10:
        return "00"+str(num)
    elif num < 100:
        return "0"+str(num)
    else:
        return str(num)

def main():

    if args['--mode'] == "list":
        show_mysql_conte(application_path)

    else:
        if args['--mode'] == "database":
            learning_corpus = retrieve_mysql_conte(format_nbr(53), Corpus.TEXT_FR.value[2], application_path, False)
        else:
            learning_corpus = samples.cinderella

        for txt in learning_corpus:
            phrases = SpacyPhrase(nlp(txt))
            phrases.preprocessing()
            phrases.handle_scenes()
            phrases.make_sentence()
            phrases.write()


if __name__ == '__main__':
    main()
