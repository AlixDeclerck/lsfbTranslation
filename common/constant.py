from enum import Enum
from collections import namedtuple

dir_separator = "/"
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

start_symbol = 0
stop_symbol = 1
pad_idx = 2

def pretty_print_hypothesis(hypothesis):
    res = str(hypothesis[0].value[0])+" "
    for i in range(1, len(hypothesis)):
        res += str(hypothesis[i].value[0])+" "
        if hypothesis[i].value[0] == str(Tag.STOP.value):
            break

    print("Model Output               : " + res)

class Tag(Enum):
    """
    Types of tags
    """
    START = "<s>"
    STOP = "</s>"
    BLANK = "<blank>"
    UNKNOWN = "<unk>"
