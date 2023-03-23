#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    dal.py testing --app-path=<file>
"""

import mysql.connector
import pandas
from docopt import docopt
import os
from common.constant import dir_separator


def data_provider(db, application_path):
    """
    A persisted data provider
    @param db: db_dev or db_test
    @return: mysql connector
    :param application_path:
    """
    local = True
    address = "localhost"
    df = pandas.read_csv(application_path+CONFIG)

    if local:
        with open(application_path+CONFIG_LOCAL) as f:
            lines = f.readlines()

        user = lines[0]
        pwd = lines[1]

    else:
        user = df["user"][0]
        pwd = df["pwd"][0]

    return mysql.connector.connect(
        host=address,
        user=user,
        password=pwd,
        database=df[db][0])


def select_mysql_datas_from(subset_type, conn):
    query_result = []
    p_sources = "p.text, p.gloss"
    query = "select "+str(p_sources)+" from PARALLEL_ITEM as p inner join ENVIRONMENT as e on p.env_type = e.envId where e.type = '"+str(subset_type)+"';"
    cur = conn.cursor()
    cur.execute(query)

    for x in cur.fetchall():
        query_result.append({"text": x[0], "gloss": x[1]})

    conn.close()

    return query_result


def display_envs(conn):
    query_result = []
    p_sources = "*"
    query = "select "+p_sources+" from ENVIRONMENT;"
    cur = conn.cursor()
    cur.execute(query)

    for x in cur.fetchall():
        query_result.append({"envs": x[0]})

    conn.close()
    print(query_result)


# --------------------------------------------- Execution ----- -----------

CONFIG = "algorithms/data_loader/config.csv"                # generic information included in the project
CONFIG_LOCAL = "algorithms/data_loader/config_local.txt"    # connection information specific to your local configuration


if __name__ == "__main__":

    args = docopt(__doc__)
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

    display_envs(data_provider("db_dev", application_path))
