from enum import Enum


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
    PUNCTUATION = "punct"
    VERB = "VERB"
    PERSON = "PROPN"


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
    Provide some functionalities to walk the Tree
    """
    def __init__(self):
        self.nodes = []

    @property
    def size(self):
        return self.__len__

    def preprocessing(self, words):
        for k, word in enumerate(words):
            if word.deprel == WordType.PUNCTUATION.value:
                words.pop(k)
            elif word.head == 0:
                new_node = WordNode(words.pop(k))
                new_node.initialize()
                self.nodes.append(new_node)

        return words

    def search_node(self, node_id):
        for node in self.nodes:
            if node.word.id == node_id:
                return node
        return None

    def add_node(self, word):
        parent_node = self.search_node(word.head)
        if not (parent_node is None):
            new_child = parent_node.add_child(word)
            new_child.initialize()
            self.nodes.append(new_child)
            return True
        return False

    def show(self, node, space):
        if node.relevant:
            print(space+node.word.text+" " +
                  (str(node.event) if len(node.event) > 0 else "") +
                  (str(node.classificator) if len(node.classificator) > 0 else "") +
                  (str(node.subject) if len(node.subject) > 0 else "") +
                  (str(node.action) if len(node.action) > 0 else ""))

        space = space+Config.DISPLAY.value
        for child in node.children:
            self.show(child, space)

    def gloss(self, node):
        res = ["", "", "", ""]

        for i in range(3):
            txt = ""
            if node.relevant:
                if i == 0 and len(node.event) > 0:
                    txt = str(res[i] + node.word.text + " ")

                if i == 1 and len(node.classificator) > 0:
                    txt = str(res[i] + node.word.text + " ")

                if i == 2 and len(node.subject) > 0:
                    txt = str(res[i] + node.word.text + " ")

                if i == 3 and len(node.action) > 0:
                    txt = str(res[i] + node.word.text + " ")

                res[i] = txt

        for i in range(3):
            if res[i] != "" and res[i] != " ":
                print(res[i])

        for child in node.children:
            self.gloss(child)

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, item_id):
        return [node.id for node in self.nodes if node.identifier is item_id]


class WordNode:
    """
    A structure for each node received from a sentence
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
        if "DET" == self.word.upos or "PRON" == self.word.upos or self.word.lemma == "de" or self.word.lemma == "que" or "ADP" == self.word.upos:
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
        node = WordNode(word)
        node.parent = self
        self.children.append(node)
        # print(f"added node {node.word.text} to {self.word.text} .. ")
        return node

    def child_number(self):
        return len(self.children)


# Hypothesis : only one phrase
def load_sentenza(doc):
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
            if vocabulary.add_node(word):
                updated_words.remove(word)

        words = updated_words
        if len(words) == 0 or i == original_word_size:
            break

        i += 1

    return vocabulary
