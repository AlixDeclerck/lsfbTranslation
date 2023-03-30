#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    inject_data.py --app-path=<file>
"""

import pandas
import dal as db
import retrieve_data as ff
from common.constant import dir_separator, EnvType, SPLIT_FACTOR, XLSX_PATH, CSV_PATH, SELECTED_DB
from docopt import docopt
import os

class ConteHandler:

    def __init__(self, application_path):
        self.application_path = application_path
        self.stakeholders_dir = application_path + XLSX_PATH
        self.learning_dir = application_path + CSV_PATH

    def convert(self):
        """
        To convert from XLSX files to CSV files
        """
        files_source = self.retrieve_xlsx_url()
        files_destination = [x.replace(".xlsx", ".csv") for x in files_source]
        for i in range(len(files_source)):
            source_path = self.stakeholders_dir+files_source[i]
            destination_path = self.learning_dir+files_destination[i]
            df = pandas.DataFrame(pandas.read_excel(source_path))
            df.to_csv(destination_path, index=False, header=True)

    def populate_db(self):
        """
        add in database (db_creation/contes)
        a new population of conte from csv files
        """
        conn = db.data_provider(SELECTED_DB, self.application_path)
        cpt = 0
        for s in self.retrieve_csv_contes():

            story_name = s.removesuffix(".csv")
            conte = ff.get_conte(self.learning_dir + s)
            for i, ln in enumerate(conte.iterrows()):

                if (i % SPLIT_FACTOR) == 0:
                    env_name = EnvType.TEST.value
                else:
                    env_name = EnvType.TRAINING.value

                print(f"--- {env_name} insertions --------------")
                text_fr = ln[1].FR
                gloss_lsf = ln[1].GLOSS_LSF
                generated = ln[1].GENERATED
                tense = ln[1].TENSE
                gloss_lsfb = ln[1].GLOSS_LSFB
                text_en = ln[1].EN
                num_line = ln[1].NUM
                print(f"[{i}] inserted {story_name} | {text_fr} | {gloss_lsf} | {generated} | {tense} | {gloss_lsfb} | {text_en} | {num_line} | {env_name} ")
                sql = "INSERT INTO PARALLEL_ITEM (story_name, FR, GLOSS_LSF, GENERATED_LSF, TENSE, GLOSS_LSFB, EN, NUM, env_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                val = (story_name, text_fr, gloss_lsf, generated, tense, gloss_lsfb, text_en, num_line, env_name)

                cur = conn.cursor()
                cur.execute(sql, val)

                conn.commit()
                cpt += 1

        print(f"{cpt} row inserted")
        conn.close()

    def retrieve_csv_contes(self):
        """
        :return: list of csv conte
        """
        actual_path = os.getcwd()
        content = []
        if os.path.exists(self.learning_dir):
            os.chdir(self.learning_dir)
            content = os.listdir()
            os.chdir(actual_path)

        return content

    def print_xlsx_data(self):
        """
        To check files in the xlsx source folder
        """
        actual_path = os.getcwd()
        if os.path.exists(self.stakeholders_dir) and os.path.exists(self.learning_dir):
            os.chdir(self.stakeholders_dir)
            content = os.listdir()
            print(content)
            os.chdir(actual_path)
            print("actual path :", os.path.realpath(__file__))

    def retrieve_xlsx_url(self):
        """
        :return: list with xlsx files url
        """
        actual_path = os.getcwd()
        content = []
        if os.path.exists(self.stakeholders_dir) and os.path.exists(self.learning_dir):
            os.chdir(self.stakeholders_dir)
            content = os.listdir()
            os.chdir(actual_path)

        return content

# ----------------------------------------------------------


if __name__ == "__main__":

    args = docopt(__doc__)
    contes = ConteHandler(os.environ['HOME']+dir_separator+args['--app-path']+dir_separator)

    # list the cvs in conte directory
    print(contes.retrieve_csv_contes())

    # uncomment to create csv files from the xlsx source folder
    # contes.convert()

    # uncomment to add csv population to database (3346 row inserted)
    # contes.populate_db()
