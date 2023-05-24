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

NUMBER_OF_TRAINING_RESULTS = 17

if __name__ == '__main__':

    # update path from given parameters
    config = load_config("../algorithms/symbolicTransformer/src/config.yaml")
    args = docopt(__doc__)
    path = os.environ['HOME'] + config["configuration_path"]["application_path"] + args['--app-path'] + config["configuration_path"]["application_path"] + "algorithms/symbolicTransformer/src/output/"

    # retrieve loss
    df = pandas.read_csv(str(path)+"learning_symbolicTransformer_french_23-05-24_B.csv")
    loss_column = df.iloc[:, [2]]
    validation_column = df.iloc[:, [0]]
    learning_rate_column = df.iloc[:, [4]]

    res = []
    i = 0
    for ln in loss_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0:5] == " Loss":
            i += 1
            if i % NUMBER_OF_TRAINING_RESULTS == 0:
                tmp = ln[0].split(" ")[4]
                res.append(float(tmp))

    res_lr = []
    j = 0
    for ln in learning_rate_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0:14] == " Learning Rate":
            j += 1
            if j % NUMBER_OF_TRAINING_RESULTS == 0:
                tmp = ln[0].split(" ")[3]
                res_lr.append(float(tmp)*10000)

    res_eval = []
    for ln in validation_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0] == "(":
            tmp = ln[0].split("(")[2].split(",")[0]
            res_eval.append(float(tmp))

    pocket_eval = []
    for ln in validation_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0] == "(":
            tmp = float(ln[0].split(",")[3].split(":")[1])
            pocket_eval.append(tmp)

    # labels
    training_set = str(validation_column.values.tolist()[0]).split(" : ")[1][1:-1]
    label_training = "training "+training_set+" "
    score = "{:.3f}".format(pocket_eval[-1])
    label_pocket = "best validation score : "+score
    label_validation = "training score"
    label_lr = "learning rate * 10 000"

    # display loss curve
    plt.plot(range(len(res_eval)), res_eval, c=str(config["graphics"]["color4"]), label=label_validation)
    plt.plot(range(len(pocket_eval)), pocket_eval, c=str(config["graphics"]["color2"]), label=label_pocket)
    plt.plot(range(len(res)), res, c=str(config["graphics"]["color1"]), label=label_training)
    plt.plot(range(len(res_lr)), res_lr, c=str(config["graphics"]["color3"]), label=label_lr)
    plt.legend()
    plt.ylabel("Kullback-Leibler divergence loss")
    plt.xlabel("epochs")
    plt.title("Experimentation 24/5/23 - smoothing 0.5 - batch size 8")
    plt.savefig('img/learning_curves_ST_2023-05-24_B.png')
    plt.show()