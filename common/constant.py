from enum import Enum
from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def pretty_print_hypothesis(hypothesis, method):
    res = str(hypothesis.value[0])+" "
    for i in range(1, len(hypothesis.value)):
        res += str(hypothesis.value[i])+" "
        if hypothesis.value[i] == str(Tag.STOP.value[0]):
            break

    print("\nModel Output ( " + method + " )     : " + str(res))
    return res

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
    BOTH = 0
    LSF = 1
    GENERATED = 2
