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
            if word.deprel == WordType.PUNCTUATION.value or word.upos == WordType.DETERMINANT.value or word.upos == "ADP": #https://universaldependencies.org/u/pos/ADP.html ADP should perhaps be filtered?
                continue
            elif word.head == 0:
                new_item = WordItem(word)
                new_item.initialize()
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
            # new_child.initialize()
            self.items.append(new_child)
            return True
        return False

    """
    show the indented list of linked glosses using head to
    display as a tree (BFS walk)
    """
    def show(self, item, space):
        if item.relevant:
            print(space + item.word.text + " " +
                  (str(item.event) if len(item.event) > 0 else "") +
                  (str(item.classificator) if len(item.classificator) > 0 else "") +
                  (str(item.subject) if len(item.subject) > 0 else "") +
                  (str(item.action) if len(item.action) > 0 else ""))

        space = space+Config.DISPLAY.value
        for child in item.children:
            self.show(child, space)

    """
    show the glosses
    """
    def gloss(self, item):
        res = ["", "", "", ""]

        for i in range(3):
            txt = ""
            if item.relevant:
                if i == 0 and len(item.event) > 0:
                    txt = str(res[i] + item.word.text + " ")

                if i == 1 and len(item.classificator) > 0:
                    txt = str(res[i] + item.word.text + " ")

                if i == 2 and len(item.subject) > 0:
                    txt = str(res[i] + item.word.text + " ")

                if i == 3 and len(item.action) > 0:
                    txt = str(res[i] + item.word.text + " ")

                res[i] = txt

        for i in range(3):
            if res[i] != "" and res[i] != " ":
                print(res[i])

        for child in item.children:
            self.gloss(child)

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
    """

    def __init__(self, word):
        self.word = word
        self.parent = None
        self.children = []
        self.event = []
        self.classificator = []
        self.subject = []
        self.action = []
        self.relevant = True
        self.unclassified = False

    def initialize(self):
        if "PRON" == self.word.upos or "ADP" == self.word.upos:
            self.relevant = False
        elif "INTJ" == self.word.upos:
            self.action.append(" with "+str(self.parent.word.text))
        elif WordType.VERB.value == self.word.upos or "AUX" == self.word.upos:
            self.action.append(SceneItem.ACTION.value+" ")
            if self.word.feats.split("|")[0].split("=")[1] == "Inf":
                self.event.append("Inf")
            else:
                self.event.append(self.word.feats.split("|")[3].split("=")[1])
        elif WordType.PERSON.value == self.word.upos:
            self.subject.append(WordType.PERSON.value+" ")
        elif "NOUN" == self.word.upos or "NUM" == self.word.upos:
            self.subject.append("NOUN")
        else:
            self.unclassified = True

    def add_child(self, word):
        item = WordItem(word)
        item.parent = self
        self.children.append(item)
        print(f"added item {item.word.text} to {self.word.text} .. ")
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
