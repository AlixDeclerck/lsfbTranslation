from enum import Enum
from collections import namedtuple

dir_separator = "/"
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class Tag(Enum):
    """
    Types of tags
    """
    START = "<s>"
    STOP = "</s>"
    BLANK = "<blank>"
    UNKNOWN = "<unk>"
