import pandas
from algorithms.data_loader.src import dal
from common.constant import SELECTED_DB, Corpus

def get_phoenix(path):
    """
    get the Phoenix csv content filtering by column name
    :param path: csv file path
    :return: phoenix dataframe
    """
    phoenix = pandas.read_csv(path, sep=',', delimiter=None, header='infer', names=None, index_col=None)
    phoenix.columns = ["gloss", "text"]
    return phoenix.filter(["text", "gloss"])

def get_conte(path):
    """
    get a conte csv content filtering by column name
    :param path: csv file path
    :return: conte dataframe
    """
    conte = pandas.read_csv(path, sep=',', delimiter=None, header='infer', names=None, index_col=None, na_values=str, keep_default_na=False)
    conte.columns = ["FR", "GLOSS_LSF", "GENERATED_LSF", "TENSE", "GLOSS_LSFB", "EN", "NUM", "GENERATED_FR", "GENERATED_EN"]
    return conte.filter(["FR", "GLOSS_LSF", "GENERATED_LSF", "TENSE", "GLOSS_LSFB", "EN", "NUM", "GENERATED_FR", "GENERATED_EN"])

def show_mysql_conte(application_path):
    """
    get a dictionary from dataset :
    :param application_path: the application path to retrieve database information
    :return: dictionary
    """
    request = "select distinct p.story_name from PARALLEL_ITEM p order by p.story_name asc;"
    pt = dal.data_provider(SELECTED_DB, application_path)
    cur = pt.cursor()
    cur.execute(request)
    for x in cur.fetchall():
        print("- "+x[0])


def retrieve_mysql_conte(conte_num, language, application_path):
    """
    get a dictionary from dataset :
    :param conte_num: a conte designed by a num
    :param language: text_fr, text_en, gloss_lsf
    :param application_path: the application path to retrieve database information
    :return: dictionary
    """
    res = []
    request = "select p.num, p."+str(language)+" from PARALLEL_ITEM p where p.story_name like '"+str(conte_num)+"%';"
    pt = dal.data_provider(SELECTED_DB, application_path)
    cur = pt.cursor()
    cur.execute(request)

    for x in cur.fetchall():
        res.append(x[1])

    pt.close()

    return res


def retrieve_mysql_datas_from(subset_type, application_path):
    """
    get a dictionary from dataset : text_fr, text_en, gloss_lsf
    :param subset_type: selected environment
    :param application_path: the application path to retrieve database information
    :return: dictionary
    """
    res = []
    request = "select p.FR, p.GENERATED_FR, p.EN, p.GENERATED_EN, p.GLOSS_LSF, p.GENERATED_LSF from PARALLEL_ITEM as p where p.env_type = '"+str(subset_type)+"';"
    pt = dal.data_provider(SELECTED_DB, application_path)
    cur = pt.cursor()
    cur.execute(request)

    for x in cur.fetchall():
        if x[0] is not None:
            corpus_fr = x[0]
        else:
            corpus_fr = x[1]

        if x[3] is not None:
            corpus_en = x[3]
        else:
            corpus_en = x[2]

        if x[4] is not None:
            corpus_glosses = x[4]
        else:
            corpus_glosses = x[5]

        res.append({
            Corpus.TEXT_FR.value[0]: corpus_fr,
            Corpus.TEXT_EN.value[0]: corpus_en,
            Corpus.GLOSS_LSF.value[0]: corpus_glosses
        })

    pt.close()

    return res
