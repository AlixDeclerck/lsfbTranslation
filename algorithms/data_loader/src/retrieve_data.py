import pandas
from algorithms.data_loader.src import dal
from common.constant import Corpus, Dialect

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

def show_mysql_conte(application_path, selected_db):
    """
    get a dictionary from dataset :
    :param application_path: the application path to retrieve database information
    :param selected_db : db_dev or db_test
    :return: dictionary
    """
    request = "select distinct p.story_name from TRANSLATION p order by p.story_name asc;"
    pt = dal.data_provider(selected_db, application_path)
    cur = pt.cursor()
    cur.execute(request)
    for x in cur.fetchall():
        print("- "+x[0])


def retrieve_mysql_conte(conte_num, language, application_path, selected_db, generated=False):
    """
    get a dictionary from dataset :
    :param conte_num: a conte designed by a num
    :param language: text_fr, text_en, gloss_lsf
    :param application_path: the application path to retrieve database information
    :param selected_db : database (db_dev or db_test)
    :param generated: False when expected translation, true when hypothesis
    :return: dictionary
    """
    if generated:
        txt_value = "p.txt_generated"
    else:
        txt_value = "p.txt"

    res = []
    request = "select p.num, "+str(txt_value)+" from TRANSLATION p where p.story_name like '%"+str(conte_num)+"%' and lang = '"+str(language)+"';"
    pt = dal.data_provider(selected_db, application_path)
    cur = pt.cursor()
    cur.execute(request)

    for x in cur.fetchall():
        res.append(x[1])

    pt.close()

    return res


def retrieve_mysql_datas_from(subset_type, application_path, selected_db, dialect_selection=0, src_multi=False, request_limit=10000):
    """
    get a dictionary from dataset : text_fr, text_en, gloss_lsf
    :param subset_type: selected environment
    :param application_path: the application path to retrieve database information
    :param selected_db : the database used (db_dev or db_test)
    :param dialect_selection : to choose which glosses are taken {0:"both", 1:"LSF", 2:"generated"}
    :param src_multi : If true we take booth generated and FR source when possible
    :param request_limit : To limit the request output
    :return: dictionary
    """
    res = []
    both_value = False

    if Dialect.LSF.value[0] == dialect_selection:
        request = "select p2.txt as txt_fr, p2.txt_generated as fr_generated, p1.txt as txt_en, p1.txt_generated as en_generated, p3.txt as txt_gloss from TRANSLATION p1 inner join TRANSLATION p2 on p1.num = p2.num and p1.story_name = p2.story_name inner join TRANSLATION p3 on p2.num = p3.num and p2.story_name = p3.story_name where p1.lang = 'EN' and p2.lang = 'FR' and p3.lang = 'GLOSS_LSF' and p1.env_type = '"+str(subset_type)+"' limit "+str(request_limit)+" ;"

    elif Dialect.GENERATED.value[0] == dialect_selection:
        request = "select p2.txt as txt_fr, p2.txt_generated as fr_generated, p1.txt as txt_en, p1.txt_generated as en_generated, p3.txt_generated as gloss_generated from TRANSLATION p1 inner join TRANSLATION p2 on p1.num = p2.num and p1.story_name = p2.story_name inner join TRANSLATION p3 on p2.num = p3.num and p2.story_name = p3.story_name where p1.lang = 'EN' and p2.lang = 'FR' and p3.lang = 'GLOSS_LSF' and p1.env_type = '"+str(subset_type)+"' limit "+str(request_limit)+" ;"

    else:
        request = "select p2.txt as txt_fr, p2.txt_generated as fr_generated, p1.txt as txt_en, p1.txt_generated as en_generated, p3.txt as txt_gloss, p3.txt_generated as gloss_generated from TRANSLATION p1 inner join TRANSLATION p2 on p1.num = p2.num and p1.story_name = p2.story_name inner join TRANSLATION p3 on p2.num = p3.num and p2.story_name = p3.story_name where p1.lang = 'EN' and p2.lang = 'FR' and p3.lang = 'GLOSS_LSF' and p1.env_type = '"+str(subset_type)+"' limit "+str(request_limit)+" ;"
        both_value = True

    pt = dal.data_provider(selected_db, application_path)
    cur = pt.cursor()
    cur.execute(request)

    for x in cur.fetchall():
        corpus_fr = []
        if src_multi and x[0] != "" and x[1] != "":
            corpus_fr.append(x[0])
            corpus_fr.append(x[1])
        elif x[0] != "":
            corpus_fr.append(x[0])  # txt_fr
        else:
            corpus_fr.append(x[1])  # fr_generated

        if x[3] != "":
            corpus_en = x[3]  # en_generated
        else:
            corpus_en = x[2]  # txt_en

        if both_value:
            if x[4] != "":
                corpus_glosses = x[4]   # txt_gloss
            else:
                corpus_glosses = x[5]   # gloss_generated
        else:
            corpus_glosses = x[4]       # txt_gloss or gloss_generated depending on the request

        for c in corpus_fr:
            res.append({
                Corpus.TEXT_FR.value[0]: c,
                Corpus.TEXT_EN.value[0]: corpus_en,
                Corpus.GLOSS_LSF.value[0]: corpus_glosses
            })

    pt.close()

    return res
