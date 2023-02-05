#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    learning_phase.py train --app-path=<file>
"""

import os

import matplotlib.pyplot as plt
import pandas
from docopt import docopt

from common.constant import dir_separator

curves_color1 = '#5BCFFA'
curves_color2 = '#F5ABB9'

if __name__ == '__main__':

    # update path from given parameters
    args = docopt(__doc__)
    path = os.environ['HOME'] + dir_separator + args['--app-path'] + dir_separator + "algorithms/symbolicTransformer/src/output/"

    # retrieve loss
    df = pandas.read_csv(str(path)+"learning_symbolicTransformer_french_23-02-05.csv")
    loss_column = df.iloc[:, [2]]

    res = []
    for ln in loss_column.values.tolist():
        if str(ln[0]) != 'nan' and ln[0][0:5] == " Loss":
            tmp = ln[0].split(" ")[4]
            res.append(tmp)

    # display loss curve
    plt.plot(range(len(res)), res, c=curves_color1, label="symbolic transformer loss")
    plt.legend()
    plt.ylabel("Learning")
    plt.gca().invert_yaxis()
    plt.savefig('img/learning_curves_ST_2023-02-05.png')
    plt.show()

