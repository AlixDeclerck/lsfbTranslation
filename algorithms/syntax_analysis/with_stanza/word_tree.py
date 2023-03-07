from constant import WordType

class WordTree:
    """
    Provide some functionalities to walk annotated glosses list
    """
    def __init__(self):
        self.items = []
        self.display = []
        self.glosses = {"event": [], "classificator": [], "subject": [], "action": [], "unclassified": [], "tense": []}

    @property
    def size(self):
        return self.__len__

    """
    remove punctuation sentenza type and 
    extract the head of a word to vocabulary
    """
    def list_preprocessing(self, words):
        filtered_words = []
        for word in words:
            if WordType.PUNCTUATION.value != word.deprel:  # or word.upos in ["DET", "PRON", "ADP"]: #https://universaldependencies.org/u/pos/ADP.html ADP should perhaps be filtered?
                filtered_words.append(word)

        return filtered_words

    """
    add a word to the glosses dictionary 
    (keeping sentence sort)
    """
    def add_gloss_to_list(self, word):

        # CLASSIFICATOR (2)
        if word.upos in ["CCONJ", "ADV"]:
            self.glosses["classificator"].append(word)
            if word.feats is not None:
                details = word.feats.split("|")
                for detail in details:
                    if detail.split("=")[0] == "Polarity":
                        self.glosses["classificator"].append(detail.split("=")[1])

        # SUBJECT (3)
        elif word.upos in ["NOUN", "NUM", "ADJ", "PROPN"]:
            self.glosses["subject"].append(word)

        # ACTION (4)
        elif word.upos in ["VERB", "AUX"]:
            self.glosses["action"].append(word)

            # tense
            details = word.feats.split("|")
            for detail in details:
                if detail.split("=")[0] == "Tense":
                    t = detail.split("=")[1]
                    self.glosses["tense"].append("["+str(t)+"]")

        # EVENT (1)
        elif word.upos in ["INTJ"]:
            self.glosses["event"].append(word)

        # NO MATCHING
        else:
            self.glosses["unclassified"].append(word)

    def list_display(self):
        res = ""
        for name, text in self.glosses.items():
            if name != "unclassified" and name != "tense" and name != "action":

                # for the : event, classificator, subject, action
                for t in text:
                    if type(t) == str:
                        continue

                    # try to find parent (perhaps go to ancestor) in unclassified
                    founded = False  # remove condition to ancestor it
                    for u in self.glosses["unclassified"]:
                        if u.head == t.id and "DET" != t.upos and not founded:
                            res += str(u.text)+" "
                            founded = True

                    # add the gloss to output string
                    res += str(t.text)+" "

            if name == "action":
                for t in text:
                    res += str(t.lemma)+" "

        print(res.upper())

    """
    search an item
    """
    def search_item(self, item_id):
        for item in self.items:
            if item.word.id == item_id:
                return item
        return None

    """
    add a word if this word have parent
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

    def __len__(self):
        return len(self.items)

    def __contains__(self, item_id):
        return [item.id for item in self.items if item.identifier is item_id]
