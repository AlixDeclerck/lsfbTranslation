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

    def add_child(self, word):
        item = WordItem(word)
        item.parent = self
        self.children.append(item)
        # print(f"added item {item.word.text} to {self.word.text} .. ")
        return item

    def child_number(self):
        return len(self.children)
