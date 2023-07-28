#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    plot_inference.py --app-path=<file>
"""
import os
import numpy
import matplotlib.pyplot as plt
import pandas
from docopt import docopt

from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import Case, d_date

case = Case.FIRST
session = "session d'analyse'"
add = "SF_"

if __name__ == '__main__':

    # CONFIG
    today = d_date()
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "common/output/"
    filename = "decoding_scores_2023-07-28.csv"  # "learning_symbolicTransformer_french_"+today+"_"+str(add)+case.value[1]+"_quicktranslations.csv"

    # RETRIEVE SCORES
    df = pandas.read_csv(str(path)+filename)
    img_precision = "img/precision_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_trigram = "img/trigram_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_bp = "img/bp_"+today+"_"+str(add)+str(case.value[1])+".png"
    title = "Inférences : "+str(session)+", cas n°"+str(case.value[0])

    scores_beam = pandas.DataFrame(df[(df['title'] == "Beam")], columns=["bleu", "bp", "trigram"])
    scores_greedy = pandas.DataFrame(df[(df['title'] == "Greedy")], columns=["bleu", "bp", "trigram"])
    scores_approx = pandas.DataFrame(df[(df['title'] == "Approximation")], columns=["bleu", "bp", "trigram"])
    number_of_scores = len(scores_approx)

    # PRECISION : n-gram extraction
    inference_method = ("Beam", "Greedy", "Approximation")
    precision_beam = pandas.DataFrame(df[(df['title'] == "Beam")], columns=["precision"])
    precision_greedy = pandas.DataFrame(df[(df['title'] == "Greedy")], columns=["precision"])
    precision_approx = pandas.DataFrame(df[(df['title'] == "Approximation")], columns=["precision"])
    beam_ngrams = []
    greedy_ngrams = []
    approx_ngrams = []
    for i in range(number_of_scores):
        beam_ngrams.append([float(x) for x in precision_beam.to_numpy()[i][0][1:-1].split(", ")])
        greedy_ngrams.append([float(x) for x in precision_greedy.to_numpy()[i][0][1:-1].split(", ")])
        approx_ngrams.append([float(x) for x in precision_approx.to_numpy()[i][0][1:-1].split(", ")])

    beam_ngrams = pandas.DataFrame(beam_ngrams)
    greedy_ngrams = pandas.DataFrame(greedy_ngrams)
    approx_ngrams = pandas.DataFrame(approx_ngrams)

    # pandas plot (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)
    df = pandas.DataFrame([
        ['Beam', beam_ngrams[0].mean(), beam_ngrams[1].mean(), beam_ngrams[2].mean(), beam_ngrams[3].mean()],
        ['Greedy', greedy_ngrams[0].mean(), greedy_ngrams[1].mean(), greedy_ngrams[2].mean(), greedy_ngrams[3].mean()],
        ['Approx', approx_ngrams[0].mean(), approx_ngrams[1].mean(), approx_ngrams[2].mean(), approx_ngrams[3].mean()]],
        columns=['ordre', 'unigram', 'bigram', '3-gram', '4-gram'])

    df.plot(x='ordre',
            kind='bar',
            stacked=False,
            title='Mesure de traduction par n-grammes',
            color=["#d1ade0", "#fec5d6", "#f8e392", "#9be1eb"]
            )

    plt.savefig(img_precision)
    plt.show()

    # TRI-GRAMS
    beam_trigram = [float(x) for x in scores_beam["trigram"]]
    greedy_trigram = [float(x) for x in scores_greedy["trigram"]]

    df = pandas.DataFrame(

        beam_trigram,

        columns=['beam'])

    df['greedy'] = greedy_trigram

    ax = df.plot.hist(bins=12, alpha=0.5)

    plt.title("score des tri-grammes")
    plt.savefig(img_trigram)
    plt.show()

    # BP
    beam_bp = [float(x) for x in scores_beam["bp"]]
    greedy_bp = [float(x) for x in scores_greedy["bp"]]

    df = pandas.DataFrame(

        beam_bp,

        columns=['beam'])

    df['greedy'] = greedy_bp

    ax = df.plot.hist(bins=12, alpha=0.5)

    plt.title("pénalité de concision")
    plt.savefig(img_bp)
    plt.show()

