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
from algorithms.data_loader.src.data_validation import duplicate_sentence_detection
from common.constant import d_date, current_session, HypothesisType

case = current_session()[0]
session = current_session()[1]
add = ""

class BiasDataDetection:
    def __init__(self):
        self.duplicate_items = duplicate_sentence_detection(config, args)
        self.approx = []
        self.beam = []
        self.greedy = []

    def filter_dataframe(self, df_to_filter, hypothesis_type):
        updated_df = []
        for index, row in df_to_filter.iterrows():
            if row["src"] in self.duplicate_items:
                self.quarantine_value(hypothesis_type, row["src"])
            else:
                updated_df.append(row)

        return pandas.DataFrame(updated_df)

    def quarantine_value(self, hypothesis_type, value):
        if HypothesisType.BEAM == hypothesis_type:
            self.beam.append(value)
        elif HypothesisType.GREEDY == hypothesis_type:
            self.greedy.append(value)
        else:
            self.approx.append(value)


def meteor_score_staffing(scores, staff):
    for i in range(len(scores)):
        if scores[i] < 0.1:
            staff[0] += 1
        elif scores[i] < 0.2:
            staff[1] += 1
        elif scores[i] < 0.3:
            staff[2] += 1
        elif scores[i] < 0.4:
            staff[3] += 1
        elif scores[i] < 0.5:
            staff[4] += 1
        elif scores[i] < 0.6:
            staff[5] += 1
        elif scores[i] < 0.7:
            staff[6] += 1
        elif scores[i] < 0.8:
            staff[7] += 1
        elif scores[i] < 0.9:
            staff[8] += 1
        else:
            staff[9] += 1

    return staff


if __name__ == '__main__':

    # CONFIG
    today = d_date()
    colors_1 = ["#d1ade0"]
    colors_2 = ["#fec5d6", "#f8e392"]
    colors_3 = ["#fec5d6", "#f8e392", "#9be1eb"]
    colors_4 = ["#d1ade0", "#fec5d6", "#f8e392", "#9be1eb"]
    labels_3 = ["Beam search", "Greedy search", "Approximation"]
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "common/output/"
    filename = "decoding_scores_"+today+"_"+str(add)+case.value[1]+".csv"
    img_precision = "img/precision_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_unigram = "img/unigram_" + today + "_" + str(add) + str(case.value[1]) + ".png"
    img_bp = "img/bp_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_distrib = "img/distribution_"+today+"_"+str(add)+str(case.value[1])+".png"
    roc_approx = "img/roc_approx_"+today+"_"+str(add)+str(case.value[1])+".png"
    roc_beam = "img/roc_beam_"+today+"_"+str(add)+str(case.value[1])+".png"
    roc_greedy = "img/roc_greedy_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_meteor = "img/meteor_"+today+"_"+str(add)+str(case.value[1])+".png"
    img_meteor_brut = "img/meteor_brut_"+today+"_"+str(add)+str(case.value[1])+".png"
    title = "Inférences : "+str(session)+", cas n°"+str(case.value[0])
    n_bins = 10

    # RETRIEVE SCORES
    df = pandas.read_csv(str(path)+filename)
    brut_score_beam = pandas.DataFrame(df[(df['title'] == "Beam")], columns=["precision", "bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length", "tp", "fp", "tn", "fn", "src", "ref", "hyp"])
    brut_score_greedy = pandas.DataFrame(df[(df['title'] == "Greedy")], columns=["precision", "bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length", "tp", "fp", "tn", "fn", "src", "ref", "hyp"])
    brut_score_approx = pandas.DataFrame(df[(df['title'] == "Approximation")], columns=["precision", "bleu", "bp", "unigram", "bigram", "trigram", "score_meteor", "hypothesis_length", "reference_length", "tp", "fp", "tn", "fn", "src", "ref", "hyp"])

    # FILTERING
    filter_bias = BiasDataDetection()
    scores_beam = filter_bias.filter_dataframe(brut_score_beam, HypothesisType.APPROX)
    scores_greedy = filter_bias.filter_dataframe(brut_score_greedy, HypothesisType.APPROX)
    scores_approx = filter_bias.filter_dataframe(brut_score_approx, HypothesisType.APPROX)
    number_of_scores = len(scores_approx)
    title_meteor = "Score METEOR pour "+str(number_of_scores)+" inférences"
    title_meteor_brut = "Score METEOR pour "+str(len(brut_score_approx))+" inférences"

    # PRECISION : n-gram extraction
    inference_method = ("Beam", "Greedy", "Approximation")
    precision_beam = pandas.DataFrame(scores_beam, columns=["precision"])
    precision_greedy = pandas.DataFrame(scores_greedy, columns=["precision"])
    precision_approx = pandas.DataFrame(scores_approx, columns=["precision"])
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
            color=colors_4)

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

    ax = df_bigram.plot.hist(bins=n_bins, alpha=0.5, color=colors_3)

    plt.title("score des unigrammes")
    plt.savefig(img_unigram)
    plt.show()

    # BP
    beam_bp = [float(x) for x in scores_beam["bp"]]
    greedy_bp = [float(x) for x in scores_greedy["bp"]]

    df_bp_beam = pandas.DataFrame(
        beam_bp,
        columns=['beam'])

    df_bp_beam['greedy'] = greedy_bp

    ax = df_bp_beam.plot.hist(bins=n_bins, alpha=0.5, color=colors_2)
    plt.xlabel('Valeur de la pénalité')
    plt.ylabel('Fréquence')

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

    scores_by_orders_2 = [score_order_beam["bigram"], score_order_greedy["bigram"], score_order_approx["bigram"]]
    scores_by_orders_3 = [score_order_beam["trigram"], score_order_greedy["trigram"], score_order_approx["trigram"]]
    scores_by_orders_4 = [score_order_beam["fourgram"], score_order_greedy["fourgram"], score_order_approx["fourgram"]]
    scores_by_orders_5 = [score_order_beam["fivegram"], score_order_greedy["fivegram"], score_order_approx["fivegram"]]

    unigram = [beam_ngrams[0], greedy_ngrams[0], approx_ngrams[0]]
    bigram = [beam_ngrams[1], greedy_ngrams[1], approx_ngrams[1]]
    trigram = [beam_ngrams[2], greedy_ngrams[2], approx_ngrams[2]]

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

    ax0.hist(scores_approx["reference_length"], n_bins, density=True, histtype='bar', color="#d1ade0")
    ax0.legend(prop={'size': 10})
    ax0.set_title('Nombre de mots de la référence')

    ax1.hist(scores_approx["hypothesis_length"], n_bins, density=True, histtype='bar', color="#9be1eb")
    ax1.legend(prop={'size': 10})
    ax1.set_title("Nbr de mots de l'approximation")

    ax2.hist(scores_beam["hypothesis_length"], n_bins, density=True, histtype='bar', color="#fec5d6")
    ax2.legend(prop={'size': 10})
    ax2.set_title("Nbr de mots de l'inférence beam")

    ax3.hist(scores_greedy["hypothesis_length"], n_bins, density=True, histtype='bar', color="#f8e392")
    ax3.legend(prop={'size': 10})
    ax3.set_title("Nbr de mots de l'inférence greedy")

    # ax0.hist(scores_by_orders_2, n_bins, density=True, histtype='bar', color=colors, label=labels)
    # ax0.legend(prop={'size': 10})
    # ax0.set_title('bigrammes des références d'ordre 2')

    # ax1.hist(scores_by_orders_3, n_bins, density=True, histtype='bar', color=colors_3)
    # ax1.legend(prop={'size': 10})
    # ax1.set_title("bigrammes des références d'ordre 3")

    # ax2.hist(scores_by_orders_4, n_bins, density=True, histtype='bar', color=colors_3)
    # ax2.legend(prop={'size': 10})
    # ax2.set_title("bigrammes des références d'ordre 4")

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

    # METEOR
    beam_scores_meteor = [float(x) for x in scores_beam["score_meteor"]]
    greedy_scores_meteor = [float(x) for x in scores_greedy["score_meteor"]]
    approx_scores_meteor = [float(x) for x in scores_approx["score_meteor"]]

    np_meteor_beam = meteor_score_staffing(scores=beam_scores_meteor, staff=numpy.zeros(n_bins))
    np_meteor_greedy = meteor_score_staffing(scores=greedy_scores_meteor, staff=numpy.zeros(n_bins))
    np_meteor_approx = meteor_score_staffing(scores=approx_scores_meteor, staff=numpy.zeros(n_bins))

    df_meteor = pandas.DataFrame(
        np_meteor_beam,
        columns=['beam'])

    df_meteor['greedy'] = np_meteor_greedy
    df_meteor['approximation'] = np_meteor_approx

    # df_meteor = pandas.DataFrame([
    #     ['Beam', df_meteor[0], df_meteor[1], df_meteor[2], df_meteor[3], df_meteor[4], df_meteor[5], df_meteor[6], df_meteor[7], df_meteor[8], df_meteor[9]],
    #     ['Greedy', df_meteor[0], df_meteor[1], df_meteor[2], df_meteor[3], df_meteor[4], df_meteor[5], df_meteor[6], df_meteor[7], df_meteor[8], df_meteor[9]],
    #     ['Approx', df_meteor[0], df_meteor[1], df_meteor[2], df_meteor[3], df_meteor[4], df_meteor[5], df_meteor[6], df_meteor[7], df_meteor[8], df_meteor[9]]],
    #     columns=['type', '<1', '<2', '<3', '<4', '<5', '<6', '<7', '<8', '<9', '<10'])

    df_meteor = pandas.DataFrame([
        ['<1', np_meteor_beam[0], np_meteor_greedy[0], np_meteor_approx[0]],
        ['<2', np_meteor_beam[1], np_meteor_greedy[1], np_meteor_approx[1]],
        ['<3', np_meteor_beam[2], np_meteor_greedy[2], np_meteor_approx[2]],
        ['<4', np_meteor_beam[3], np_meteor_greedy[3], np_meteor_approx[3]],
        ['<5', np_meteor_beam[4], np_meteor_greedy[4], np_meteor_approx[4]],
        ['<6', np_meteor_beam[5], np_meteor_greedy[5], np_meteor_approx[5]],
        ['<7', np_meteor_beam[6], np_meteor_greedy[6], np_meteor_approx[6]],
        ['<8', np_meteor_beam[7], np_meteor_greedy[7], np_meteor_approx[7]],
        ['<9', np_meteor_beam[8], np_meteor_greedy[8], np_meteor_approx[8]],
        ['<10', np_meteor_beam[9], np_meteor_greedy[9], np_meteor_approx[9]]],
        columns=['type', 'beam', 'greedy', 'approximation']
    )

    df_meteor.plot(x='type',
                        kind='bar',
                        stacked=False,
                        title='Mesure METEOR par inférences',
                        color=colors_3)

    plt.xlabel('Score')
    plt.ylabel("Nombre d'occurences")
    plt.title(title_meteor)
    plt.savefig(img_meteor)
    plt.show()

    # METEOR BRUT
    beam_scores_meteor_brut = [float(x) for x in brut_score_beam["score_meteor"]]
    greedy_scores_meteor_brut = [float(x) for x in brut_score_greedy["score_meteor"]]
    approx_scores_meteor_brut = [float(x) for x in brut_score_approx["score_meteor"]]

    np_meteor_beam_brut = meteor_score_staffing(scores=beam_scores_meteor_brut, staff=numpy.zeros(n_bins))
    np_meteor_greedy_brut = meteor_score_staffing(scores=greedy_scores_meteor_brut, staff=numpy.zeros(n_bins))
    np_meteor_approx_brut = meteor_score_staffing(scores=approx_scores_meteor_brut, staff=numpy.zeros(n_bins))

    df_meteor_brut = pandas.DataFrame([
        ['<1', np_meteor_beam_brut[0], np_meteor_greedy_brut[0], np_meteor_approx_brut[0]],
        ['<2', np_meteor_beam_brut[1], np_meteor_greedy_brut[1], np_meteor_approx_brut[1]],
        ['<3', np_meteor_beam_brut[2], np_meteor_greedy_brut[2], np_meteor_approx_brut[2]],
        ['<4', np_meteor_beam_brut[3], np_meteor_greedy_brut[3], np_meteor_approx_brut[3]],
        ['<5', np_meteor_beam_brut[4], np_meteor_greedy_brut[4], np_meteor_approx_brut[4]],
        ['<6', np_meteor_beam_brut[5], np_meteor_greedy_brut[5], np_meteor_approx_brut[5]],
        ['<7', np_meteor_beam_brut[6], np_meteor_greedy_brut[6], np_meteor_approx_brut[6]],
        ['<8', np_meteor_beam_brut[7], np_meteor_greedy_brut[7], np_meteor_approx_brut[7]],
        ['<9', np_meteor_beam_brut[8], np_meteor_greedy_brut[8], np_meteor_approx_brut[8]],
        ['<10', np_meteor_beam_brut[9], np_meteor_greedy_brut[9], np_meteor_approx_brut[9]]],
        columns=['type', 'beam', 'greedy', 'approximation']
    )

    df_meteor_brut.plot(x='type',
                   kind='bar',
                   stacked=False,
                   title='Mesure METEOR BRUT par inférences',
                   color=colors_3)

    plt.xlabel('Score')
    plt.ylabel("Nombre d'occurences")
    plt.title(title_meteor_brut)
    plt.savefig(img_meteor_brut)
    plt.show()
