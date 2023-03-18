from constant import Config
from phrases import Phrase

def load_sentenza(doc):
    """
    Load a (one phrase) sentence
    """
    if len(doc.sentences) == 0:
        return Config.EMPTY_PHRASE
    else:
        phrase = doc.sentences[0].words

    vocabulary = Phrase()

    words = vocabulary.list_preprocessing(phrase)
    for word in words:
        vocabulary.add_gloss_to_list(word)

    return vocabulary
