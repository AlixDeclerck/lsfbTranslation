#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    launcher.py contes --mode=string --app-path=<file>
"""

import os
import spacy
from docopt import docopt
from algorithms.syntax_analysis.with_spacy.phrases import SpacyPhrase
from data.other_conte import samples
from algorithms.data_loader.src.retrieve_data import retrieve_mysql_conte, show_mysql_conte
from common.constant import Corpus
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config

txt_samples = [
    "La perfection réside dans l'amour que l'on porte.",
    "Chaque nouvelle journée est (une instance) unique comme toutes les autres.",
    "Au 21° siècle, la définition du mot femme s'étend comme le coeur des hommes s'ouvre.",
    "Je considère Edward Snowden comme un héros et il n'a trouvé asile qu'en Russie. Cela me terrifie!",
]

# nlp : the doc object
nlp = spacy.load("fr_core_news_sm")

# --------------------------------------------------------------------------

def format_nbr(num):
    if num < 10:
        return "00"+str(num)
    elif num < 100:
        return "0"+str(num)
    else:
        return str(num)

def main():
    """
    A parameter define a mode (list or databases) to know where we read the file
    The configuration keep a persist-approx boolean that define the output:
    True for populate the database, false to display on screen
    :return: approximations
    """

    args = docopt(__doc__)
    config = load_config(os.environ['HOME']+"/"+args['--app-path']+"/algorithms/symbolicTransformer/src/config.yaml")
    application_path = os.environ['HOME']+config["configuration_path"]["application_path"]+args['--app-path']+config["configuration_path"]["application_path"]
    output_only_glosses = True

    if args['--mode'] == "list":
        show_mysql_conte(application_path, config["configuration_path"]["selected_db"])
        for txt in samples.cinderella:
            phrases = SpacyPhrase(nlp(txt))
            phrases.preprocessing()
            phrases.handle_scenes()
            phrases.grammar_handler()
            phrases.handle_output(glosses=output_only_glosses)

    else:
        if args['--mode'] == "database":
            learning_corpus = retrieve_mysql_conte(
                conte_num=format_nbr(55),
                language=Corpus.TEXT_FR.value[2],
                application_path=str(application_path),
                selected_db=config["configuration_path"]["selected_db"],
                generated=False)

            res = []
            for txt in learning_corpus:
                phrases = SpacyPhrase(nlp(txt))
                phrases.preprocessing()
                phrases.handle_scenes()
                phrases.grammar_handler()
                res.append(phrases.handle_output(glosses=output_only_glosses))

            if output_only_glosses:
                for r in res:
                    print(r)


def approximate_phrases(conte, cfg):
    res = []
    for txt in conte:
        phrases = SpacyPhrase(nlp(txt))
        phrases.preprocessing()
        phrases.handle_scenes()
        phrases.grammar_handler()
        res.append(phrases.handle_output(glosses=cfg["inference_decoding"]["persist-approx"]))

    return res


if __name__ == '__main__':
    main()
