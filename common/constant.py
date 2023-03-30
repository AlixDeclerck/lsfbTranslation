from enum import Enum
from collections import namedtuple


# todo in config file :
color1 = '#FF99EE'
color2 = '#7799EE'
SELECTED_DB = "db_dev"
SPLIT_FACTOR = 7
XLSX_PATH = "data/other_conte/xlsx/"
CSV_PATH = "data/conte/csv/"
dir_separator = "/"  # dir separator already there
# ---

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def pretty_print_hypothesis(hypothesis):
    res = str(hypothesis.value[0])+" "
    for i in range(1, len(hypothesis.value)):
        res += str(hypothesis.value[i])+" "
        if hypothesis.value[i] == str(Tag.STOP.value[0]):
            break

    print("Model Output               : " + res)

class Tag(Enum):
    """
    Types of tags
    """
    START = "<s>", 0
    STOP = "</s>", 1
    BLANK = "<blank>", 2
    UNKNOWN = "<unk>", 3

class EnvType(Enum):
    """
    Types of environments
    The enum values are used both
    in dataset directories and database tables
    """
    TEST = "test"         # model comparing in decoding_phase.py
    TRAINING = "train"    # training in learning_phase.py
    # VALIDATION = "val"  # model metrics not define here : batch split val from train

class Corpus(Enum):
    """
    Corpus items
    The enum gives the dataframe index
    for each item coming from database
    """
    TEXT_FR = "text_fr", 0
    TEXT_EN = "text_en", 1
    GLOSS_LSF = "gloss_lsf", 2
