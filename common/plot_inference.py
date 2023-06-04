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
from statistics import stdev
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
    if value > 100:
        return True, value
    else:
        return False, value


case = Case.THIRD
session = "session 03"
add = "S3_"

if __name__ == '__main__':

    # update path from given parameters
    today = d_date()
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "algorithms/symbolicTransformer/src/output/"

    # retrieve score
    df = pandas.read_csv(str(path)+"learning_symbolicTransformer_french_"+today+"_"+str(add)+case.value[1]+"_quicktranslations.csv")
    filename = "img/inference_scores_"+today+"_"+str(add)+str(case.value[1])+".png"
    title = "Inférences : "+str(session)+", cas n°"+str(case.value[0])
    inference_result_title = df.iloc[:, [0]].values.tolist()
    inference_result_data = df.iloc[:, [1]].values.tolist()
    beam_scores = []
    beam_outliers = []
    greedy_scores = []
    greedy_outliers = []

    for i in range(len(inference_result_title)):
        target = i + 1
        if str(inference_result_title[i][0])[15:19] == "beam":

            normalized = normalize_result(float(re.sub('-',  '', inference_result_data[target][0]).replace(" ", "")))
            if normalized[0]:
                beam_outliers.append(normalized[1])
            else:
                beam_scores.append(normalized[1])

        elif str(inference_result_title[i][0])[15:21] == "greedy":

            normalized = normalize_result(float(re.sub('-',  '', inference_result_data[target][0]).replace(" ", "")))
            if normalized[0]:
                greedy_outliers.append(normalized[1])
            else:
                greedy_scores.append(normalized[1])

    # processing outliers
    total_beam_outliers = len(beam_outliers)
    total_greedy_outliers = len(greedy_outliers)

    # processing average
    total_beam_score = 0
    total_greedy_score = 0
    for i in range(len(beam_scores)):
        total_beam_score += beam_scores[i]

    for i in range(len(greedy_scores)):
        total_greedy_score += greedy_scores[i]

    beam_mean = total_beam_score / float(len(beam_scores))
    greedy_mean = total_greedy_score / float(len(greedy_scores))

    beam_score_mean = []
    greedy_score_mean = []

    for i in range(len(beam_scores)):
        beam_score_mean.append(beam_mean)

    for i in range(len(greedy_scores)):
        greedy_score_mean.append(greedy_mean)

    std_beam_score = str(round(stdev(beam_scores), 2))
    std_greedy_score = str(round(stdev(greedy_scores), 2))
    label_beam_score = "beam scores (écart-type : "+std_beam_score+", nbr. v. aberrante(s) : "+str(total_beam_outliers)+")"
    label_greedy_score = "greedy scores (écart-type : "+std_greedy_score+", nbr. v. aberrante(s) : "+str(total_greedy_outliers)+")"
    avg_beam_score = "beam en moyenne : "+str(round(beam_mean, 2))
    avg_greedy_score = "greedy en moyenne : "+str(round(greedy_mean, 2))
    N = config["learning_config"]["output_max_words"]
    label_x = "nombre de phrases"
    label_y = "score BLEU exprimé en % avec N="+str(N)

    fig = plt.figure()
    x_beam = range(len(beam_scores))
    x_greedy = range(len(greedy_scores))
    plt.errorbar(x_beam, beam_scores, c=str(config["graphics"]["color1"]), label=label_beam_score)
    plt.errorbar(x_beam, beam_score_mean, c=str(config["graphics"]["color2"]), dashes=[1, 2], label=avg_beam_score)
    plt.errorbar(x_greedy, greedy_scores, c=str(config["graphics"]["color3"]), label=label_greedy_score)
    plt.errorbar(x_greedy, greedy_score_mean, c=str(config["graphics"]["color4"]), dashes=[1, 2], label=avg_greedy_score)
    plt.ylim(0, 150)
    plt.legend()
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.title(title)
    plt.savefig(filename)
    plt.show()
