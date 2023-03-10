#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    _main_.py train --app-path=<file>
"""

import dal as db
import retrieve_data as ff
from common.constant import dir_separator
from docopt import docopt
import os

def populate_db_from_phoenix(subset_type):
    """
    add in database a new population based on a translated version of the phoenix
    :param subset_type: train, dev, test
    """
    conn = db.data_provider("db_dev", application_path)
    mirrored_env = db.env_provider(subset_type)
    path = "../../../data/phoenix_fr/phoenix."+mirrored_env+".corpus.csv"
    fawkes = ff.get_phoenix(path)

    cpt = 0
    for i, ln in enumerate(fawkes.iterrows()):
        print(f"--- {mirrored_env} insertions --------------")
        gloss = ln[1].display_gloss_from_inner_tree
        text = ln[1].text
        print(f"[{i}] inserted in {mirrored_env} : {gloss} / {text}")
        sql = "INSERT INTO PARALLEL_ITEM (text, gloss, env_type) VALUES (%s, %s, %s)"
        val = (text, gloss, subset_type)

        cur = conn.cursor()
        cur.execute(sql, val)

        conn.commit()
        cpt += 1

    print(f"{cpt} row inserted")
    conn.close()

# ----------------------------------------------------------


if __name__ == "__main__":

    args = docopt(__doc__)
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

    for env in db.EnvType:
        print(db.env_provider(env.value[0]))

        # !! following line will insert the texts + glosses from provided files to selected database
        # populate_db_from_phoenix(env.value[0])
