#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    plot_ST_learning.py --app-path=<file>
"""

import os

import matplotlib.pyplot as plt
import pandas
from docopt import docopt
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config

NUMBER_OF_TRAINING_RESULTS = 16

if __name__ == '__main__':

    # update path from given parameters
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "algorithms/symbolicTransformer/src/output/"

    # retrieve loss
    df = pandas.read_csv(str(path)+"learning_symbolicTransformer_french_23-05-12.csv")
    loss_column = df.iloc[:, [2]]
    validation_column = df.iloc[:, [0]]

    res = []
    i = 0
    for ln in loss_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0:5] == " Loss":
            i += 1
            if i % NUMBER_OF_TRAINING_RESULTS == 0:
                tmp = ln[0].split(" ")[4]
                res.append(float(tmp))

    res_eval = []
    for ln in validation_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0] == "(":
            tmp = ln[0].split("(")[2].split(",")[0]
            res_eval.append(float(tmp))

    # display loss curve
    plt.plot(range(len(res)), res, c=str(config["graphics"]["color1"]), label="learning phase - glosses training")
    plt.plot(range(len(res_eval)), res_eval, c=str(config["graphics"]["color2"]), label="learning phase - glosses validation")
    plt.legend()
    plt.ylabel("Kullback-Leibler divergence loss")
    plt.xlabel("epochs")
    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    plt.savefig('img/learning_curves_ST_2023-05-12.png')
    plt.show()
