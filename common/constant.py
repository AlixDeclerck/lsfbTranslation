from enum import Enum

dir_separator = "/"


class Tag(Enum):
    """
    Types of tags
    """
    START = "<s>"
    STOP = "</s>"
    BLANK = "<blank>"
    UNKNOWN = "<unk>"
