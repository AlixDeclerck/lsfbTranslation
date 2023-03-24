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
    conte.columns = ["FR", "GLOSS_LSF", "GENERATED", "TENSE", "GLOSS_LSFB", "EN"]
    # return conte.filter(["FR", "GLOSS_LSF", "GENERATED", "TENSE", "GLOSS_LSFB", "EN"])
    return conte

def retrieve_mysql_datas_from(subset_type, application_path):
    """
    get a dictionary from dataset : text_fr, text_en, gloss_lsf
    :param subset_type: selected environment
    :param application_path: the application path to retrieve database informations
    :return: dictionary
    """
    res = []
    request = "select p.FR, p.EN, p.GLOSS_LSF from PARALLEL_ITEM as p where p.env_type = '"+str(subset_type)+"';"
    pt = dal.data_provider(SELECTED_DB, application_path)
    cur = pt.cursor()
    cur.execute(request)

    for x in cur.fetchall():
        res.append({
            Corpus.TEXT_FR.value[0]: x[0],
            Corpus.TEXT_EN.value[0]: x[1],
            Corpus.GLOSS_LSF.value[0]: x[2]
        })

    pt.close()

    return res
