import unicodedata

from common.constant import Tag
from scene_construction import SubPhrase


class SpacyPhrase:
    """
    Provide some functionalities to walk annotated glosses list
    """
    def __init__(self, txt):
        self.raw_txt = txt          # a phrase with nlp spacy format
        self.filtered = []          # filtered words during preprocessing
        self.phrases = []           # phrase after pre_preprocessing
        self.scene = []             # a list of sub_phrases
        self.tokens = []            # a list of tokens to display
        self.tenses = []            # a list of tenses (parallel with tokens)

    """
    constructing glosses from raw_text 
    making a preprocessing into glosses dictionary:
    - filtering {DETERMINANT, SYMBOL} and stop words
    """
    def preprocessing(self):
        for token in self.raw_txt:
            if token.text in ["°"]:
                self.filtered.append(token)  # stop words
            elif token.pos_ in ["PUNCT"] and token.text not in [".", ";", ",", ":"]:
                self.filtered.append(token)
            elif token.pos_ in ["DET", "SYM"]:
                self.filtered.append(token)
            else:
                self.phrases.append(token)

    """
    do a scenes segmentation by punctuation signs into sub phrases
    """
    def handle_scenes(self):
        sub_phrase = SubPhrase()
        for token in self.phrases:
            if "PUNCT" == token.pos_ or "et" == token.text:
                if sub_phrase.__len__() > 0:
                    self.scene.append(sub_phrase)
                    sub_phrase = SubPhrase()
            else:
                sub_phrase.add_word(token)
        if sub_phrase.__len__() > 0:
            self.scene.append(sub_phrase)

    """
    using basic grammatical scene construction steps 
    """
    def make_sentence(self):
        self.tokens = []
        self.tenses = []
        for sub_phrase in self.scene:
            for s in sub_phrase.event:
                self.tokens.append(s)
            for s in sub_phrase.classificator:
                self.tokens.append(s)
            for s in sub_phrase.subject:
                self.tokens.append(s)
            for s in sub_phrase.action:
                self.tokens.append(s)
            for s in sub_phrase.tense:
                self.tenses.append(s)

    """
    create a sentence from tokens and print it
    """
    def write(self):
        # print(f"1 phrase with {self.__len__()} sub phrases")  # sanity check
        print(self.raw_txt)

        # tenses ()
        if len(self.tenses) < 1:
            tenses = str(Tag.UNKNOWN.value)
        else:
            tenses = "".join([str(x) for x in self.tenses if len(x) > 0])

        # display tokens and tenses
        if len(self.tokens) < 1:
            print(str(Tag.UNKNOWN.value)+"|"+tenses)
        else:
            res = ""

            for t in self.tokens:
                if not isinstance(t, str):
                    res += t.text+" "
                else:
                    res += t+" "

            formated_res = "".join([x for x in unicodedata.normalize("NFKD", res).upper() if not unicodedata.combining(x)])
            print(formated_res+"|"+tenses)  # the | sign permit easy integration into csv

        # lines separator
        print("-----")

    def __len__(self):
        return len(self.scene)

