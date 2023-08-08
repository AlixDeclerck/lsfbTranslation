#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    inject_data.py --app-path=<file>
"""

import pandas
import datetime
import dal as db
import retrieve_data as ff
from data_validation import duplicate_sentence_detection
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import EnvType, Corpus, Tag, Hypothesis
from common.metrics.bleu import processing_bleu_score
from algorithms.syntax_analysis.with_spacy.launcher import approximate_phrases
from algorithms.symbolicTransformer.src.functionnal.data_preparation import retrieve_conte_dataset
from docopt import docopt
import os
from common.constant import Dialect

class ConteHandler:

    def __init__(self, application_path, xlsx_path, csv_path, sf, gl, selected_db):
        self.application_path = application_path
        self.stakeholders_dir = application_path + xlsx_path
        self.learning_dir = application_path + csv_path
        self.split_factor = sf
        self.gloss_limit = gl
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

        sql = "delete from TRANSLATION"
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

        is_persist = config["inference_decoding"]["persist-approx"]

        if is_persist:
            if conte.FR[0] != "":
                corpus = conte.FR.values
            else:
                corpus = conte.GENERATED_FR.values

            lsf_approx = approximate_phrases(corpus, config)
            if len(list(conte.iterrows())) != len(lsf_approx):
                print("Sanity check fail : No congruence with approximation size")
                is_persist = False

        else:
            lsf_approx = None

        cpt_test = 0

        for i, ln in enumerate(conte.iterrows()):

            # LSF approximation
            if is_persist:
                data = lsf_approx[i].split("|")
                generated_lsf = data[0]
                generated_tense = data[1]
                now = datetime.datetime.utcnow()
                generation_date = now.strftime('%Y-%m-%d %H:%M:%S')
            else:
                generated_lsf = ln[1].GENERATED_LSF
                generated_tense = ln[1].TENSE
                generation_date = None

            # environment definition
            regularize_gloss = 0 < self.gloss_limit and (self.gloss_limit < len(ln[1].GLOSS_LSF.split(" ")) or self.gloss_limit < len(ln[1].GENERATED_LSF.split(" ")))
            if not regularize_gloss and (i % int(self.split_factor)) == 0:
                env_name = EnvType.TEST.value
                cpt_test += 1
            else:
                env_name = EnvType.TRAINING.value

            print(f"--- {env_name} insertions --------------")

            # processing parallels by language
            for lang in languages:
                if lang[0] == Corpus.TEXT_FR.value[2]:
                    self.parallel_insertion(conn, ln[1].NUM, lang[0], i, story_name, ln[1].FR, ln[1].GENERATED_FR, generated_tense, generation_date, env_name, config["inference_decoding"]["output_max_words"])
                elif lang[0] == Corpus.TEXT_EN.value[2]:
                    self.parallel_insertion(conn, ln[1].NUM, lang[0], i, story_name, ln[1].EN, ln[1].GENERATED_EN, "", generation_date, env_name, config["inference_decoding"]["output_max_words"])
                elif lang[0] == Corpus.GLOSS_LSF.value[2]:
                    self.parallel_insertion(conn, ln[1].NUM, lang[0], i, story_name, ln[1].GLOSS_LSF, generated_lsf, "", generation_date, env_name, config["inference_decoding"]["output_max_words"])

            cpt += 1
            print(f"=== {cpt_test} values added in test set ===")
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

    @staticmethod
    def parallel_insertion(conn, num_line, lang, i, story_name, text, generated, tense, today_date, env_name, max_output):
        if text != "" and generated != "":
            reference, hypothesis = assemble_txt_bleu(text, generated)
            score = processing_bleu_score(reference, hypothesis, output_max=max_output, display=True)
        else:
            score = 0

        print(f"[{i}] insert {story_name}| {lang} | {text} | {generated} | {tense} | {score} | {today_date} | {num_line} | {env_name} ")
        sql = "INSERT INTO TRANSLATION (story_name, num, lang, txt, txt_generated, tense, score, txt_generation_date, env_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (story_name, num_line, lang, text, generated, tense, score, today_date, env_name)

        cur = conn.cursor()
        cur.execute(sql, val)
        conn.commit()

def assemble_txt_bleu(ref, hyp):
    return [Tag.START.value[0]] + ref.split(" ") + [Tag.STOP.value[0]], [Hypothesis(value=[Tag.START.value[0]] + hyp.split(" ") + [Tag.STOP.value[0]], score=None)][0]


def show_bleu_score(ref, hyp):
    reference, hypothesis = assemble_txt_bleu(ref, hyp)
    print(f"ref : {ref}")
    print(f"hyp : {hyp}")
    processing_bleu_score(
        reference,
        hypothesis,
        # output_max=config["inference_decoding"]["output_max_words"],
        output_max=3,
        shrink=False,
        display=True)

def display_duplicate_sentence():
    items = duplicate_sentence_detection(config=config, args=args)
    for i, item in enumerate(items):
        print("duplicate "+str(i)+" : "+item)

# ----------------------------------------------------------


if __name__ == "__main__":

    # CONFIGURATION
    args = docopt(__doc__)
    config = load_config("../../symbolicTransformer/src/config.yaml")

    contes = ConteHandler(
        application_path=os.environ['HOME']+config['configuration_path']['application_path']+args['--app-path']+config['configuration_path']['application_path'],
        xlsx_path=config['configuration_path']['xlsx_path'],
        csv_path=config['configuration_path']['csv_path'],
        sf=config['learning_config']['test_division'],
        gl=config['learning_config']['max_test_gloss_size'],
        selected_db=config['configuration_path']['selected_db'])

    # list the cvs in conte directory
    # print(contes.retrieve_csv_contes())

    # uncomment to create csv files from the xlsx source folder
    # contes.convert()

    # uncomment to add csv population to database (5802 phrases inserted)
    # contes.populate_db_from_csv()

    # show bleu score
    # show_bleu_score(ref="BLANCHE NEIGE FENETRE REGARDER DIT BONJOUR", hyp="DAME FENETRE REGARDER")

    # find duplicate txt in test set
    display_duplicate_sentence()
