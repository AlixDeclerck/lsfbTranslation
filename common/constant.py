from enum import Enum
from collections import namedtuple
from datetime import datetime

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def d_date():
    today_date = datetime.today()
    today_year = str(today_date.year)[-2:]
    today_month = "0"+str(today_date.month) if today_date.month < 10 else str(today_date.month)
    today_day = "0"+str(today_date.day) if today_date.day < 10 else str(today_date.day)
    return today_year+"-"+today_month+"-"+today_day

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
    TEXT_FR = "text_fr", 0, "FR"
    TEXT_EN = "text_en", 1, "EN"
    GLOSS_LSF = "gloss_lsf", 2, "GLOSS_LSF"

class Dialect(Enum):
    BOTH = 0, "Gloses générées + traductions LSF"
    LSF = 1, "traductions LSF"
    GENERATED = 2, "Gloses générées"

class Case(Enum):
    """
        To choose the file where inference is
        and write the title
    """
    FIRST = "1", "A"
    SECOND = "2", "B"
    THIRD = "3", "C"
    FOURTH = "4", "D"
    FIFTH = "5", "E"
    SIXTH = "6", "F"
    SEVEN = "7", "G"
    EIGHT = "8", "H"
    NINE = "9", "I"
    TEN = "10", "J"
    ELEVEN = "11", "K"
    TWELVE = "12", "L"
    THIRTEEN = "13", "M"
    FOURTEEN = "14", "N"
    FIFTEEN = "15", "O"

class SubCase(Enum):
    """
        To choose the file where inference is and write the title
        We might use a sub_case only on the first experiment
    """
    FIRST = "1, sous cas 1", "A1"
    SECOND = "1, sous cas 2", "A2"

class AttentionType(Enum):
    """
        Configuration to display attentions matrix
    """
    ENCODER = "Encoder"
    DECODER_SELF = "Decoder self"
    DECODER_SRC = "Decoder src"

class HypothesisType(Enum):
    """
        Types of hypothesis
    """
    BEAM = "Beam search decoding"
    GREEDY = "Greedy decoding"
    APPROX = "Approximation"

def current_session():
    return Case.FIFTH, "session 04"
