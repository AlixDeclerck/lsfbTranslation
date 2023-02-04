import pandas
from algorithms.data_loader.src import dal


def display_string(path):
    """
    A gross version to filter the Phoenix's csv content
    """
    df = pandas.read_csv(path)
    for i in df.index:
        line = str(df.values[i])
        ln_gloss = line.split("|")[5]
        ln_text = line.split("|")[6]
        print(f"ln {i} : {ln_text} / {ln_gloss}")


def get_phoenix(path):
    """
    get the Phoenix csv content filtering by column name
    :param path: csv file path
    :return: phoenix dataframe
    """
    phoenix = pandas.read_csv(path, sep=',', delimiter=None, header='infer', names=None, index_col=None)
    phoenix.columns = ["gloss", "text"]
    return phoenix.filter(["text", "gloss"])


def retrieve_mysql_datas_from(subset_type, application_path):
    res = []
    request = "select p.text, p.gloss from PARALLEL_ITEM as p inner join ENVIRONMENT as e on p.env_type = e.envId where e.type = '"+str(subset_type)+"';"
    pt = dal.data_provider("db_dev", application_path)
    cur = pt.cursor()
    cur.execute(request)

    for x in cur.fetchall():
        res.append({"src": x[0], "tgt": x[1]})

    pt.close()

    return res
