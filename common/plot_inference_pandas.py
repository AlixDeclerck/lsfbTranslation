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
from common.constant import Case, d_date, current_session

case = current_session()
session = "session d'analyse"
add = ""

if __name__ == '__main__':

    # CONFIG
    today = d_date()
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "common/output/"
    filename = "decoding_scores_"+today+"_"+str(add)+case.value[1]+".csv"

    # RETRIEVE SCORES
    df = pandas.read_csv(str(path)+filename)
    img_precision = "img/precision_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_trigram = "img/trigram_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_bp = "img/bp_"+today+"_"+str(add)+str(case.value[1])+".png"
    title = "Inférences : "+str(session)+", cas n°"+str(case.value[0])

    scores_beam = pandas.DataFrame(df[(df['title'] == "Beam")], columns=["bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length"])
    scores_greedy = pandas.DataFrame(df[(df['title'] == "Greedy")], columns=["bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length"])
    scores_approx = pandas.DataFrame(df[(df['title'] == "Approximation")], columns=["bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length"])
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
    df_ngrams_bars = pandas.DataFrame([
        ['Beam', beam_ngrams[0].mean(), beam_ngrams[1].mean(), beam_ngrams[2].mean(), beam_ngrams[3].mean()],
        ['Greedy', greedy_ngrams[0].mean(), greedy_ngrams[1].mean(), greedy_ngrams[2].mean(), greedy_ngrams[3].mean()],
        ['Approx', approx_ngrams[0].mean(), approx_ngrams[1].mean(), approx_ngrams[2].mean(), approx_ngrams[3].mean()]],
        columns=['ordre', 'unigram', 'bigram', '3-gram', '4-gram'])

    df_ngrams_bars.plot(x='ordre',
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
    approx_trigram = [float(x) for x in scores_approx["trigram"]]

    df = pandas.DataFrame(
        beam_trigram,
        columns=['beam'])

    df['greedy'] = greedy_trigram

    # ax = df.plot.hist(bins=12, alpha=0.5)
    #
    # plt.title("score des tri-grammes")
    # plt.savefig(img_trigram)
    # plt.show()

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

    df_ngrams_bars.plot(x='ordre', kind='bar', stacked=False, title='Mesure de traduction par n-grammes', color=["#d1ade0", "#fec5d6", "#f8e392", "#9be1eb"])

    # score by length
    score_order_approx = {"bigram": [], "trigram": [], "fourgram": [], "fivegram": [], "sixgram": [], "others": []}
    score_order_beam = {"bigram": [], "trigram": [], "fourgram": [], "fivegram": [], "sixgram": [], "others": []}
    score_order_greedy = {"bigram": [], "trigram": [], "fourgram": [], "fivegram": [], "sixgram": [], "others": []}
    key = "unigram"
    for k in range(0, len(scores_approx)):
        if int(scores_approx["reference_length"].tolist()[k]) == 2:
            score_order_approx["bigram"].append(scores_approx[key].tolist()[k])
            score_order_beam["bigram"].append(scores_beam[key].tolist()[k])
            score_order_greedy["bigram"].append(scores_greedy[key].tolist()[k])
        elif int(scores_approx["reference_length"].tolist()[k]) == 3:
            score_order_approx["trigram"].append(scores_approx[key].tolist()[k])
            score_order_beam["trigram"].append(scores_beam[key].tolist()[k])
            score_order_greedy["trigram"].append(scores_greedy[key].tolist()[k])
        elif int(scores_approx["reference_length"].tolist()[k]) == 4:
            score_order_approx["fourgram"].append(scores_approx[key].tolist()[k])
            score_order_beam["fourgram"].append(scores_beam[key].tolist()[k])
            score_order_greedy["fourgram"].append(scores_greedy[key].tolist()[k])
        elif int(scores_approx["reference_length"].tolist()[k]) == 5:
            score_order_approx["fivegram"].append(scores_approx[key].tolist()[k])
            score_order_beam["fivegram"].append(scores_beam[key].tolist()[k])
            score_order_greedy["fivegram"].append(scores_greedy[key].tolist()[k])
        elif int(scores_approx["reference_length"].tolist()[k]) == 6:
            score_order_approx["sixgram"].append(scores_approx[key].tolist()[k])
            score_order_beam["sixgram"].append(scores_beam[key].tolist()[k])
            score_order_greedy["sixgram"].append(scores_greedy[key].tolist()[k])
        else:
            score_order_approx["others"].append(scores_approx[key].tolist()[k])
            score_order_beam["others"].append(scores_beam[key].tolist()[k])
            score_order_greedy["others"].append(scores_greedy[key].tolist()[k])

    n_bins = 10
    x_old = numpy.random.randn(1000, 3)
    x = [beam_trigram, greedy_trigram, approx_trigram]
    meteor = [
        [float(x) for x in scores_beam["score_meteor"]],
        [float(x) for x in scores_greedy["score_meteor"]],
        [float(x) for x in scores_approx["score_meteor"]],
    ]

    scores_by_orders_2 = [score_order_beam["bigram"], score_order_greedy["bigram"], score_order_approx["bigram"]]
    scores_by_orders_3 = [score_order_beam["trigram"], score_order_greedy["trigram"], score_order_approx["trigram"]]
    scores_by_orders_4 = [score_order_beam["fourgram"], score_order_greedy["fourgram"], score_order_approx["fourgram"]]
    scores_by_orders_5 = [score_order_beam["fivegram"], score_order_greedy["fivegram"], score_order_approx["fivegram"]]

    unigram = [beam_ngrams[0], greedy_ngrams[0], approx_ngrams[0]]
    bigram = [beam_ngrams[1], greedy_ngrams[1], approx_ngrams[1]]
    trigram = [beam_ngrams[2], greedy_ngrams[2], approx_ngrams[2]]

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    colors = ["#fec5d6", "#f8e392", "#9be1eb"]
    labels = ["Beam search", "Greedy search", "Approximation"]

    ax0.hist(scores_approx["reference_length"], n_bins, density=True, histtype='bar', color="#fec5d6")
    ax0.legend(prop={'size': 10})
    ax0.set_title('distribution des références')

    # ax0.hist(scores_by_orders_2, n_bins, density=True, histtype='bar', color=colors, label=labels)
    # ax0.legend(prop={'size': 10})
    # ax0.set_title('bigrammes')

    # ax0.hist(meteor, n_bins, density=True, histtype='bar', color=colors, label=labels)
    # ax0.legend(prop={'size': 10})
    # ax0.set_title('meteor scores')

    ax1.hist(scores_by_orders_3, n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax1.legend(prop={'size': 10})
    ax1.set_title('trigrammes')

    ax2.hist(scores_by_orders_4, n_bins, density=True, histtype='bar', color=colors)
    ax2.legend(prop={'size': 10})
    ax2.set_title('quatregrammes')

    ax3.hist(scores_by_orders_5, n_bins, density=True, histtype='bar', color=colors)
    ax3.legend(prop={'size': 10})
    ax3.set_title('cinqgrammes')

    fig.tight_layout()
    plt.show()
