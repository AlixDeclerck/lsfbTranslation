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
from common.constant import d_date, current_session

case = current_session()
session = "session 02"
add = ""

if __name__ == '__main__':

    # CONFIG
    today = d_date()
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "common/output/"
    filename = "decoding_scores_"+today+"_"+str(add)+case.value[1]+".csv"

    colors_1 = ["#d1ade0"]
    colors_2 = ["#fec5d6", "#f8e392"]
    colors_3 = ["#fec5d6", "#f8e392", "#9be1eb"]
    colors_4 = ["#d1ade0", "#fec5d6", "#f8e392", "#9be1eb"]
    labels_3 = ["Beam search", "Greedy search", "Approximation"]

    # RETRIEVE SCORES
    df = pandas.read_csv(str(path)+filename)
    img_precision = "img/precision_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_unigram = "img/unigram_" + today + "_" + str(add) + str(case.value[1]) + ".png"
    img_bp = "img/bp_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_distrib = "img/distribution_"+today+"_"+str(add)+str(case.value[1])+".png"
    roc_approx = "img/roc_approx_"+today+"_"+str(add)+str(case.value[1])+".png"
    roc_beam = "img/roc_beam_"+today+"_"+str(add)+str(case.value[1])+".png"
    roc_greedy = "img/roc_greedy_"+today+"_"+str(add)+str(case.value[1])+".png"
    title = "Inférences : "+str(session)+", cas n°"+str(case.value[0])

    scores_beam = pandas.DataFrame(df[(df['title'] == "Beam")], columns=["bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length", "tp", "fp", "tn", "fn"])
    scores_greedy = pandas.DataFrame(df[(df['title'] == "Greedy")], columns=["bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length", "tp", "fp", "tn", "fn"])
    scores_approx = pandas.DataFrame(df[(df['title'] == "Approximation")], columns=["bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length", "tp", "fp", "tn", "fn"])
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
            color=colors_4
            )

    plt.savefig(img_precision)
    plt.show()

    # NGRAMS
    beam_scores_ngram = [float(x) for x in scores_beam["unigram"]]
    greedy_scores_ngram = [float(x) for x in scores_greedy["unigram"]]
    approx_scores_ngram = [float(x) for x in scores_approx["unigram"]]

    df_bigram = pandas.DataFrame(
        beam_scores_ngram,
        columns=['beam'])

    df_bigram['greedy'] = greedy_scores_ngram
    df_bigram['approximation'] = approx_scores_ngram

    ax = df_bigram.plot.hist(bins=12, alpha=0.5, color=colors_3)

    plt.title("score des unigrammes")
    plt.savefig(img_unigram)
    plt.show()

    # BP
    beam_bp = [float(x) for x in scores_beam["bp"]]
    greedy_bp = [float(x) for x in scores_greedy["bp"]]

    df = pandas.DataFrame(
        beam_bp,
        columns=['beam'])

    df['greedy'] = greedy_bp

    ax = df.plot.hist(bins=12, alpha=0.5, color=colors_2)

    plt.title("pénalité de concision")
    plt.savefig(img_bp)
    plt.show()

    # SCORE BY LENGTH
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

    ax0.hist(scores_approx["reference_length"], n_bins, density=True, histtype='bar', color=colors_1)
    ax0.legend(prop={'size': 10})
    ax0.set_title('distribution des références')

    # ax0.hist(scores_by_orders_2, n_bins, density=True, histtype='bar', color=colors, label=labels)
    # ax0.legend(prop={'size': 10})
    # ax0.set_title('bigrammes des références d'ordre 2')

    ax1.hist(scores_by_orders_3, n_bins, density=True, histtype='bar', color=colors_3)
    ax1.legend(prop={'size': 10})
    ax1.set_title("bigrammes des références d'ordre 3")

    ax2.hist(scores_by_orders_4, n_bins, density=True, histtype='bar', color=colors_3)
    ax2.legend(prop={'size': 10})
    ax2.set_title("bigrammes des références d'ordre 4")

    ax3.hist(meteor, n_bins, density=True, histtype='bar', color=colors_3, label=labels_3)
    ax3.legend(prop={'size': 10})
    ax3.set_title('meteor scores')

    # ax3.hist(scores_by_orders_5, n_bins, density=True, histtype='bar', color=colors_3)
    # ax3.legend(prop={'size': 10})
    # ax3.set_title("bigrammes des références d'ordre 5")

    fig.tight_layout()
    plt.savefig(img_distrib)
    plt.show()

    # ROC curves functions
    def plot_confusion_matrix(scores, roc_title, roc_img):
        matrix = numpy.array([[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]])
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title(roc_title)
        plt.colorbar()
        plt.xlabel('Hypothèse')
        plt.ylabel('Référence')
        plt.xticks([0, 1], ['Négatif', 'Positif'])
        plt.yticks([0, 1], ['Négatif', 'Positif'])
        plt.savefig(roc_img)
        plt.show()

    # ROC curves approx
    total_approx_scores = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(len(scores_approx)):
        total_approx_scores["tp"] += len(list(scores_approx["tp"])[i].split(", "))-1
        total_approx_scores["fp"] += len(list(scores_approx["fp"])[i].split(", "))-1
        total_approx_scores["fn"] += len(list(scores_approx["fn"])[i].split(", "))-1
        total_approx_scores["tn"] += len(list(scores_approx["tn"])[i].split(", "))-1

    plot_confusion_matrix(total_approx_scores, "Matrice de Confusion approximation", roc_approx)

    # ROC curves beam
    total_beam_scores = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(len(scores_beam)):
        total_beam_scores["tp"] += len(list(scores_beam["tp"])[i].split(", "))-1
        total_beam_scores["fp"] += len(list(scores_beam["fp"])[i].split(", "))-1
        total_beam_scores["fn"] += len(list(scores_beam["fn"])[i].split(", "))-1
        total_beam_scores["tn"] += len(list(scores_beam["tn"])[i].split(", "))-1

    plot_confusion_matrix(total_beam_scores, "Matrice de Confusion beam", roc_beam)

    # ROC curves greedy
    total_greedy_scores = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(len(scores_greedy)):
        total_greedy_scores["tp"] += len(list(scores_greedy["tp"])[i].split(", "))-1
        total_greedy_scores["fp"] += len(list(scores_greedy["fp"])[i].split(", "))-1
        total_greedy_scores["fn"] += len(list(scores_greedy["fn"])[i].split(", "))-1
        total_greedy_scores["tn"] += len(list(scores_greedy["tn"])[i].split(", "))-1

    plot_confusion_matrix(total_greedy_scores, "Matrice de Confusion greedy", roc_greedy)
