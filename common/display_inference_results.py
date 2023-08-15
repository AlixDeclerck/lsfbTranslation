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

    precision1_mean_beam = beam_ngrams[0].mean()
    precision2_mean_beam = beam_ngrams[1].mean()

    precision1_mean_greedy = greedy_ngrams[0].mean()
    precision2_mean_greedy = greedy_ngrams[1].mean()

    precision1_mean_approx = approx_ngrams[0].mean()
    precision2_mean_approx = approx_ngrams[1].mean()

    print("score unigram on the beam : ", precision1_mean_beam)
    print("score bigram on the beam : ", precision2_mean_beam)

    print("score unigram on the greedy : ", precision1_mean_greedy)
    print("score bigram on the greedy : ", precision2_mean_greedy)

    print("score unigram on the approx : ", precision1_mean_approx)
    print("score bigram on the approx : ", precision2_mean_approx)
