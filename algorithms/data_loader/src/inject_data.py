#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    inject_data.py --app-path=<file>
"""

import pandas
import dal as db
import retrieve_data as ff
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import EnvType, Corpus
from docopt import docopt
import os

class ConteHandler:

    def __init__(self, application_path, xlsx_path, csv_path, sf, selected_db):
        self.application_path = application_path
        self.stakeholders_dir = application_path + xlsx_path
        self.learning_dir = application_path + csv_path
        self.split_factor = sf
        self.selected_db = selected_db

    def convert(self):
        """
        To convert from XLSX files to CSV files
        """
        files_source = self.retrieve_xlsx_url()
        files_destination = [x.replace(".xlsx", ".csv") for x in files_source]
        for i in range(len(files_source)):
            source_path = self.stakeholders_dir+files_source[i]
            destination_path = self.learning_dir+files_destination[i]
            read_file = pandas.read_excel(source_path)
            df = pandas.DataFrame(read_file)
            df.to_csv(destination_path, index=False, header=True)

    def populate_db_from_csv(self):
        """
        add in database (db_creation/contes)
        a new population of conte from csv files
        """
        conn = db.data_provider(self.selected_db, self.application_path)

        sql = "delete from PARALLEL_ITEM "
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        cur = conn.cursor()

        request = "select * from LANG"
        cur.execute(request)
        languages = []
        for x in cur.fetchall():
            languages.append(x)

        cpt = 0
        for s in self.retrieve_csv_contes():

            story_name = s.removesuffix(".csv")
            print("insert : "+story_name)
            conte = ff.get_conte(self.learning_dir + s)
            cpt = self.populate_parallels(conn, cpt, story_name, conte, languages)

        print(f"{cpt} phrases inserted")
        conn.close()

    def populate_parallels(self, conn, cpt, story_name, conte, languages):

        for i, ln in enumerate(conte.iterrows()):

            if (i % int(self.split_factor)) == 0:
                env_name = EnvType.TEST.value
            else:
                env_name = EnvType.TRAINING.value

            print(f"--- {env_name} insertions --------------")

            for lang in languages:
                if lang[0] == Corpus.TEXT_FR.value[2]:
                    self.parallel_insertion(conn, ln[1].NUM, lang[0], i, story_name, ln[1].FR, ln[1].GENERATED_FR, ln[1].TENSE, env_name)
                elif lang[0] == Corpus.TEXT_EN.value[2]:
                    self.parallel_insertion(conn, ln[1].NUM, lang[0], i, story_name, ln[1].EN, ln[1].GENERATED_EN, "", env_name)
                elif lang[0] == Corpus.GLOSS_LSF.value[2]:
                    self.parallel_insertion(conn, ln[1].NUM, lang[0], i, story_name, ln[1].GLOSS_LSF, ln[1].GENERATED_LSF, "", env_name)

            cpt += 1
        return cpt

    @staticmethod
    def parallel_insertion(conn, num_line, lang, i, story_name, text, generated, tense, env_name):

        print(f"[{i}] inserted {story_name}| {lang} | {text} | {generated} | {tense} | {num_line} | {env_name} ")
        sql = "INSERT INTO PARALLEL_ITEM (story_name, num, lang, txt, txt_generated, tense, score, env_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (story_name, num_line, lang, text, generated, tense, 0, env_name)

        cur = conn.cursor()
        cur.execute(sql, val)

        conn.commit()

    def populate_db(self, conn, cpt, story_name, conte):

        for i, ln in enumerate(conte.iterrows()):

            if (i % int(self.split_factor)) == 0:
                env_name = EnvType.TEST.value
            else:
                env_name = EnvType.TRAINING.value

            print(f"--- {env_name} insertions --------------")
            text_fr = ln[1].FR
            gloss_lsf = ln[1].GLOSS_LSF
            generated_lsf = ln[1].GENERATED_LSF
            tense = ln[1].TENSE
            gloss_lsfb = ln[1].GLOSS_LSFB
            text_en = ln[1].EN
            num_line = ln[1].NUM
            generated_fr = ln[1].GENERATED_FR
            generated_en = ln[1].GENERATED_EN
            print(f"[{i}] inserted {story_name} | {text_fr} | {gloss_lsf} | {generated_lsf} | {tense} | {gloss_lsfb} | {text_en} | {num_line} | {generated_fr} | {generated_en} | {env_name} ")
            sql = "INSERT INTO PARALLEL_ITEM (story_name, FR, GLOSS_LSF, GENERATED_LSF, TENSE, GLOSS_LSFB, EN, NUM, GENERATED_FR, GENERATED_EN, env_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (story_name, text_fr, gloss_lsf, generated_lsf, tense, gloss_lsfb, text_en, num_line, generated_fr, generated_en, env_name)

            cur = conn.cursor()
            cur.execute(sql, val)

            conn.commit()
            cpt += 1
        return cpt

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

    # CONFIGURATION
    config = load_config("../../symbolicTransformer/src/config.yaml")

    args = docopt(__doc__)
    contes = ConteHandler(
        os.environ['HOME']+config['application_path']+args['--app-path']+config['application_path'],
        config['configuration_path']['xlsx_path'], config['configuration_path']['csv_path'],
        config['learning']['split_factor'], config['configuration_path']['selected_db'])

    # list the cvs in conte directory
    print(contes.retrieve_csv_contes())

    # uncomment to create csv files from the xlsx source folder
    # contes.convert()

    # uncomment to add csv population to database (3346 row inserted)
    # contes.populate_db_from_csv()
