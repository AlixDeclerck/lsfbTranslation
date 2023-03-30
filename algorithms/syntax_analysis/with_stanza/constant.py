from enum import Enum

class Display(Enum):
    """
    Asked functionality
    """
    SHOW_LIST = "list"
    GLOSSES = "glosses"


class Config(Enum):
    """
    General configuration
    """
    DISPLAY = "   "
    EMPTY_PHRASE = "Empty sentenza"


class WordType(Enum):
    """
    Configuration that is used to infer the glosses
    """
    PUNCTUATION = "punct"   # deprel
    VERB = "VERB"           # ?
    PERSON = "PROPN"        # ?
    DETERMINANT = "DET"     # upos


class SceneItem(Enum):
    """
    Scene description
    """
    EVENT = "Event/Tense"
    CLASSIFICATOR = "Classificator/Place"
    SUBJECT = "Subject/Personage"
    ACTION = "Action/Verb"
