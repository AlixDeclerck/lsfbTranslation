#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    plot_inference.py --app-path=<file>
"""
import re
import os
import matplotlib.pyplot as plt
import pandas
from common.constant import Case, d_date
from docopt import docopt
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config

def normalize_result(value):
    """
    Regarding the chosen N when we process BLEU score
    we might have some extreme numbers, we want that number having a constant value
    to avoid different scale during plot
    :param value : a value to normalize
    :return: a float between 0 and 100
    """
    if value > 100 or value < 1:
        return normalized_value
    else:
        return value


case = Case.SECOND
normalized_value = 0.05

if __name__ == '__main__':

    # update path from given parameters
    today = d_date()
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "algorithms/symbolicTransformer/src/output/"

    # retrieve score
    N = config["inference_decoding"]["number_of_inferences"]
    df = pandas.read_csv(str(path)+"learning_symbolicTransformer_french_"+today+"_"+case.value[1]+"_quicktranslations.csv")
    filename = "img/inference_scores_"+today+"_"+str(case.value[1])+".png"
    title = "Inférences : cas n°"+str(case.value[0]+" ("+today+")")
    inference_result_title = df.iloc[:, [0]].values.tolist()
    inference_result_data = df.iloc[:, [1]].values.tolist()
    beam_scores = []
    greedy_scores = []

    for i in range(len(inference_result_title)):
        target = i + 1
        if str(inference_result_title[i][0])[15:19] == "beam":
            beam_scores.append(normalize_result(float(re.sub('-',  '', inference_result_data[target][0]).replace(" ", ""))))
        elif str(inference_result_title[i][0])[15:21] == "greedy":
            greedy_scores.append(normalize_result(float(re.sub('-',  '', inference_result_data[target][0]).replace(" ", ""))))

    # mean
    total_beam_score = 0
    total_greedy_score = 0
    for i in range(len(beam_scores)):
        total_beam_score += beam_scores[i]
        total_greedy_score += greedy_scores[i]

    beam_mean = total_beam_score / float(N)
    greedy_mean = total_greedy_score / float(N)

    beam_score_mean = []
    greedy_score_mean = []

    for i in range(len(beam_scores)):
        beam_score_mean.append(beam_mean)
        greedy_score_mean.append(greedy_mean)

    fig = plt.figure()
    x = range(len(beam_scores))
    plt.errorbar(x, beam_scores, c=str(config["graphics"]["color1"]), label='beam scores')
    plt.errorbar(x, greedy_scores, c=str(config["graphics"]["color3"]), label='greedy scores')
    plt.errorbar(x, beam_score_mean, c=str(config["graphics"]["color2"]), dashes=[1, 2], label='beam scores en moyenne')
    plt.errorbar(x, greedy_score_mean, c=str(config["graphics"]["color4"]), dashes=[1, 2], label='greedy scores en moyenne')
    plt.legend()
    plt.ylabel("score BLEU * 100")
    plt.xlabel("phrases")
    plt.title(title)
    plt.savefig(filename)
    plt.show()
