#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    conte_loader.py train --app-path=<file>
"""

import pandas
import dal as db
import retrieve_data as ff
from common.constant import dir_separator, EnvType, XLSX_PATH, CSV_PATH, SELECTED_DB
from docopt import docopt
import os

def populate_db(environment_enum, stories):
    """
    add in database (db_creation/contes)
    a new population of conte from csv files
    """
    env_name = environment_enum.value[0]
    conn = db.data_provider(SELECTED_DB, application_path)
    dataset_path = application_path+environment_enum.value[1]

    cpt = 0
    for s in stories:

        story_name = s.removesuffix(".csv")
        conte = ff.get_conte(dataset_path+s)
        for i, ln in enumerate(conte.iterrows()):
            print(f"--- {env_name} insertions --------------")
            text_fr = ln[1].FR
            gloss_lsf = ln[1].GLOSS_LSF
            generated = ln[1].GENERATED
            tense = ln[1].TENSE
            gloss_lsfb = ln[1].GLOSS_LSFB
            text_en = ln[1].EN
            print(f"[{i}] inserted {story_name} | {text_fr} | {gloss_lsf} | {generated} | {tense} | {gloss_lsfb} | {text_en} | {env_name} ")
            sql = "INSERT INTO PARALLEL_ITEM (story_name, FR, GLOSS_LSF, GENERATED_LSF, TENSE, GLOSS_LSFB, EN, env_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (story_name, text_fr, gloss_lsf, generated, tense, gloss_lsfb, text_en, env_name)

            cur = conn.cursor()
            cur.execute(sql, val)

            conn.commit()
            cpt += 1

    print(f"{cpt} row inserted")
    conn.close()


def convert():
    files_source = retrieve_xlsx_url()
    files_destination = [x.replace(".xlsx", ".csv") for x in files_source]

    for i in range(len(files_source)):
        source_path = application_path+XLSX_PATH+files_source[i]
        destination_path = application_path+CSV_PATH+files_destination[i]
        df = pandas.DataFrame(pandas.read_excel(source_path))
        df.to_csv(destination_path, index=False, header=True)


def print_xlsl_data():
    actual_path = os.getcwd()
    source_path = application_path+XLSX_PATH
    destination_path = application_path+CSV_PATH

    if os.path.exists(source_path) and os.path.exists(destination_path):
        os.chdir(source_path)
        content = os.listdir()
        print(content)
        os.chdir(actual_path)
        print("actual path :", os.path.realpath(__file__))


def retrieve_xlsx_url():
    actual_path = os.getcwd()
    source_path = application_path+XLSX_PATH
    destination_path = application_path+CSV_PATH

    content = []
    if os.path.exists(source_path) and os.path.exists(destination_path):
        os.chdir(source_path)
        content = os.listdir()
        os.chdir(actual_path)

    return content


def retrieve_contes_names(environment_enum):
    actual_path = os.getcwd()
    csv_path = application_path+environment_enum.value[1]

    content = []
    if os.path.exists(csv_path):
        os.chdir(csv_path)
        content = os.listdir()
        os.chdir(actual_path)

    return content

# ----------------------------------------------------------


if __name__ == "__main__":

    args = docopt(__doc__)
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

    # To convert from XLSX files to CSV files
    # convert()

    # Add csv population to database
    for env in EnvType:
        populate_db(env, retrieve_contes_names(env))
