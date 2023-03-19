
class SubPhrase:
    def __init__(self):
        self.event = []
        self.classificator = []
        self.subject = []
        self.action = []
        self.tense = []
        self.unclassified = []
        self.items_number = 0

    def add_word(self, token):
        if token.pos_ in ["AUX"]:
            self.tense.append(token.morph.get("Tense")[0])

        if token.pos_ in ["ADV", "NUM"]:  # EVENT ("ADP", ?)
            if "Neg" in token.morph.get("Polarity"):
                self.event.append("PAS")
            else:
                self.event.append(token)

        if token.pos_ in ["PART"]:  # CLASSIFICATOR
            self.classificator.append(token)

        if token.pos_ in ["NOUN", "PROPN", "ADJ"]:  # SUBJECT (""PRON"
            self.subject.append(token)

        if token.pos_ in ["VERB"]:  # ACTION
            self.action.append(token.lemma_)
            self.tense.append(token.morph.get("Tense")[0])

        else:  # UNCLASSIFIED
            self.unclassified.append(token)

        self.items_number += 1

    def __len__(self):
        return self.items_number
