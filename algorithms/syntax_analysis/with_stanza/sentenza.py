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

    # BY DEPENDENCIES
    # words = vocabulary.preprocessing(phrase)
    # original_word_size = len(words)
    # i = 0
    # while True:
    #     updated_words = words.copy()
    #     for k, word in enumerate(words):
    #         if vocabulary.add_item(word):
    #             updated_words.remove(word)
    #
    #     words = updated_words
    #     if len(words) == 0 or i == original_word_size:
    #         break
    #
    #     i += 1

    # BY LIST
    words = vocabulary.list_preprocessing(phrase)
    for word in words:
        vocabulary.add_gloss_to_list(word)

    return vocabulary
