from constant import Config
from word_tree import WordTree

def load_sentenza(doc):
    """
    Load a (one phrase) sentence
    """
    if len(doc.sentences) == 0:
        return Config.EMPTY_PHRASE
    else:
        phrase = doc.sentences[0].words

    vocabulary = WordTree()

    words = vocabulary.list_preprocessing(phrase)
    for word in words:
        vocabulary.add_gloss_to_list(word)

    return vocabulary
