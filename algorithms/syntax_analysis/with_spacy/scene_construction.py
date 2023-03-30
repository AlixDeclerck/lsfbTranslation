tense_morph = "Tense"
negation_morph = "Neg"
polarity_morph = "Polarity"
negation_c1 = "PAS"
negation_c2 = "PLUS"

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
            self.tense.append(token.morph.get(tense_morph))

        if token.pos_ in ["ADV", "NUM"]:  # EVENT ("ADP", ?)
            if negation_morph in token.morph.get(polarity_morph):
                self.event.append(negation_c1)
            else:
                self.event.append(token)

        if token.pos_ in ["PART"]:  # CLASSIFICATOR
            self.classificator.append(token)

        if token.pos_ in ["NOUN", "PROPN", "ADJ"]:  # SUBJECT (""PRON"
            self.subject.append(token)

        if token.pos_ in ["VERB"]:  # ACTION
            self.action.append(token.lemma_)
            self.tense.append(token.morph.get(tense_morph))

        else:  # UNCLASSIFIED
            self.unclassified.append(token)

        self.items_number += 1

    def __len__(self):
        return self.items_number
