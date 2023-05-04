tense_morph = "Tense"
negation_morph = "Neg"
polarity_morph = "Polarity"
negation_c1 = "PAS"
negation_c2 = "PLUS"
pp_morph = "VerbForm"
pp_morph_value = "Part"

class SubPhrase:
    def __init__(self):
        self.content = []
        self.action = []
        self.tense = []
        self.unclassified = []
        self.items_number = 0

    def add_word(self, token):
        if token.pos_ in ["AUX"]:
            self.tense.append(token.morph.get(tense_morph))

        elif token.pos_ in ["NUM"]:  # EVENT ("ADP", ?)
            if negation_morph in token.morph.get(polarity_morph):
                # todo: PAS / PLUS / ...
                # self.event.append(negation_c1) # "ADV" !!
                self.content.append(token)
            else:
                self.content.append(token)

        elif token.pos_ in ["PART"]:  # CLASSIFICATOR
            self.content.append(token)

        elif token.pos_ in ["NOUN", "PROPN", "ADJ", "ADV"]:  # SUBJECT (""PRON"
            if len(token.text) > 1 and (len(token.text) > 2 or "'" not in token.text):
                self.content.append(token)

        elif token.pos_ in ["VERB"]:  # ACTION
            if pp_morph_value in token.morph.get(pp_morph):
                self.action.append(token.text)
            else:
                self.action.append(token.lemma_)

            self.tense.append(token.morph.get(tense_morph))

        else:  # UNCLASSIFIED
            self.unclassified.append(token)

        self.items_number += 1

    def __len__(self):
        return self.items_number
