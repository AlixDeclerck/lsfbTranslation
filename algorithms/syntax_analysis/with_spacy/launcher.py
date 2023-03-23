#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    launcher.py parsing --app-path=<file>
"""

import os
import spacy
from docopt import docopt
from phrases import SpacyPhrase
from data.conte import samples

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

def main():
    # learning_corpus = retrieve_mysql_datas_from(EnvType.DEV, application_path)

    for txt in samples.cinderella:
        phrases = SpacyPhrase(nlp(txt))
        phrases.preprocessing()
        phrases.handle_scenes()
        phrases.make_sentence()
        phrases.write()


if __name__ == '__main__':
    main()
