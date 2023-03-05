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


class WordTree:
    """
    Provide some functionalities to walk annotated glosses list
    """
    def __init__(self):
        self.items = []
        self.glosses = {"event": [], "classificator": [], "subject": [], "action": [], "unclassified": []}
        self.display = []

    @property
    def size(self):
        return self.__len__

    """
    remove punctuation sentenza type and 
    extract the head of a word to vocabulary
    """
    def preprocessing(self, words):
        updated_words = []
        for k, word in enumerate(words):
            if WordType.PUNCTUATION.value == word.deprel or word.upos in [WordType.DETERMINANT.value, "PRON", "ADP"]: #https://universaldependencies.org/u/pos/ADP.html ADP should perhaps be filtered?
                continue
            elif word.head == 0:
                new_item = WordItem(word)
                new_item.pre_init()
                self.items.append(new_item)
            else:
                updated_words.append(word)

        return updated_words

    """
    search if this word have a parent
    if yes add the item to the parent return true
    otherwise return false
    """
    def add_item(self, word):
        parent_item = self.search_item(word.head)
        if not (parent_item is None):
            new_child = parent_item.add_child(word)
            new_child.pre_init()
            self.items.append(new_child)
            return True
        return False

    """
    show the indented list of linked glosses using head to
    display as a BFS walk
    """
    def show(self, item, space):
        print(space + item.word.text + " " +
              (str(item.event) if len(item.event) > 0 else "") +
              (str(item.classificator) if len(item.classificator) > 0 else "") +
              (str(item.subject) if len(item.subject) > 0 else "") +
              (str(item.action) if len(item.action) > 0 else ""))

        space = space+Config.DISPLAY.value
        for child in item.children:
            self.show(child, space)

    def add_orphans(self):
        for item in self.items:
            if item.word.text not in self.display:
                self.display.append(item.word.text)

    def write(self):

        # if item.tense is not None:
        #     item.tense += self.display[0]

        res = ""
        for d in self.display:
            res += d+" "

        print(res.upper())

    """
    show the glosses
    """
    def display_gloss_from_inner_tree(self, item):
        res = ["", "", "", ""]

        for i in range(3):
            txt = ""

            if i == 0 and len(item.event) > 0:
                txt = str(res[i] + item.word.text)

            if i == 1 and len(item.classificator) > 0:
                txt = str(res[i] + item.word.text)
                # if "Neg" in item.classificator:
                #     txt = str(res[i] + "[NEG]")
                # else:
                #     txt = str(res[i] + item.word.text)

            if i == 2 and len(item.subject) > 0:
                txt = str(res[i] + item.word.text)

            if i == 3 and len(item.action) > 0:
                txt = str(res[i] + item.word.text)

            res[i] = txt

        for i in range(3):
            if res[i] != "" and res[i] != " ":
                self.display.append(str(res[i]))

        # recursive call
        for child in item.children:
            self.display_gloss_from_inner_tree(child)

    """
    search an item
    """
    def search_item(self, item_id):
        for item in self.items:
            if item.word.id == item_id:
                return item
        return None

    def __len__(self):
        return len(self.items)

    def __contains__(self, item_id):
        return [item.id for item in self.items if item.identifier is item_id]


class WordItem:
    """
    An object for each item received from a sentence
    Use Stanza's doc file
    Provide inner navigation

    Initialize provide a kind of prior knowledge based on step [1..4] to construct a visual scene

    """

    def __init__(self, word):
        self.word = word
        self.parent = None
        self.children = []
        self.event = []
        self.classificator = []
        self.subject = []
        self.action = []
        self.unclassified = False
        self.tense = ""

    def pre_init(self):
        # ACTION (4)
        if self.word.upos in ["INTJ"]:
            self.action.append(" with "+str(self.parent.word.text))

        # ACTION (4)
        elif self.word.upos in [WordType.VERB.value, "AUX"]:
            self.action.append(SceneItem.ACTION.value+" ")

            # EVENT (1)
            details = self.word.feats.split("|")
            for detail in details:
                if detail.split("=")[0] == "Tense":
                    t = detail.split("=")[1]
                    self.tense += "["+str(t)+"]"

        # SUBJECT (3)
        elif WordType.PERSON.value == self.word.upos:
            self.subject.append(WordType.PERSON.value+" ")

        elif self.word.upos in ["NOUN", "NUM", "ADJ"]:
            self.subject.append(self.word.upos+" ")

        # CLASSIFICATOR (2)
        elif self.word.upos in ["CCONJ", "ADV"]:
            self.classificator.append(self.word.upos+" ")
            if self.word.feats is not None:
                details = self.word.feats.split("|")
                for detail in details:
                    if detail.split("=")[0] == "Polarity":
                        self.classificator.append(detail.split("=")[1])

        # NO MATCHING
        else:
            self.unclassified = True

    def add_child(self, word):
        item = WordItem(word)
        item.parent = self
        self.children.append(item)
        # print(f"added item {item.word.text} to {self.word.text} .. ")
        return item

    def child_number(self):
        return len(self.children)


def load_sentenza(doc):
    """
    Load a (one phrase) sentence
    """
    if len(doc.sentences) == 0:
        return Config.EMPTY_PHRASE
    else:
        phrase = doc.sentences[0].words

    vocabulary = WordTree()
    words = vocabulary.preprocessing(phrase)
    original_word_size = len(words)

    i = 0
    while True:
        updated_words = words.copy()
        for k, word in enumerate(words):
            if vocabulary.add_item(word):
                updated_words.remove(word)

        words = updated_words
        if len(words) == 0 or i == original_word_size:
            break

        i += 1

    return vocabulary
