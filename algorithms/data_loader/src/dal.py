from enum import Enum
import mysql.connector
import pandas


class EnvType(Enum):
    """
    Types of environments
    """
    TRAIN = 1, "train"
    DEV = 2, "dev"
    TEST = 3, "test"


def env_provider(subset_type):
    for env in EnvType:
        if env.value[0] == subset_type:
            return env.value[1]
        elif env.value[1] == subset_type:
            return env.value[0]

    return None


def data_provider(db, application_path):
    """
    A persisted data provider
    @param db: db_dev or db_test
    @return: mysql connector
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


# --------------------------------------------- tests purposes ------------


def query_strange(conn):  # todo
    query_result = []
    query = "select p.text, p.gloss, e.type from PARALLEL_ITEM as p inner join ENVIRONMENT as e on p.env_type = e.envId where p.text LIKE '%august%' ;"

    cur = conn.cursor()
    cur.execute(query)

    for x in cur.fetchall():
        query_result.append({"text": x[0], "gloss": x[1]})

    conn.close()

    return query_result


def insert_row(subset_type, conn, text, gloss):
    env = env_provider(subset_type)

    sql = "INSERT INTO PARALLEL_ITEM (text, gloss, env_type) VALUES (%s, %s, %s)"
    val = (text, gloss, env)

    cur = conn.cursor()
    cur.execute(sql, val)

    conn.commit()


def delete_row(conn, text):
    val = [text]
    sql = "DELETE FROM PARALLEL_ITEM WHERE text = %s"
    # sql = "DELETE FROM PARALLEL_ITEM WHERE %s LIKE text;"
    cur = conn.cursor()
    cur.execute(sql, val)

    conn.commit()


# --------------------------------------------- Execution ----- -----------


CONFIG = "algorithms/data_loader/config.csv"                # generic information included in the project
CONFIG_LOCAL = "algorithms/data_loader/config_local.txt"    # connection information specific to your local configuration


if __name__ == "__main__":
    txt_insert = "text_content"
    gloss_insert = "glosses_content"

    print(select_mysql_datas_from("train", data_provider("db_dev")))
    # insert_row("train", data_provider("db_dev"), txt_insert, gloss_insert)
    # delete_row(data_provider(), txt_insert)
