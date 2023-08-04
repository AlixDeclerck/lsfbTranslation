from unidecode import unidecode
import string

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def process_confusion_dict(source, reference, hypothesis):
    """
    Return the true_positives, false_positives, false_negatives, true_negatives
    :param source: text to translate
    :param reference: a good translation
    :param hypothesis: an inferred translation
    :return: dictionary with confusion focused list of items
    """
    reference_tokens = set(remove_punctuation(unidecode(reference).casefold()).split())
    hypothesis_tokens = set(remove_punctuation(unidecode(hypothesis).casefold()).split())
    source_tokens = set(remove_punctuation(unidecode(source).casefold()).split())

    true_positives = reference_tokens.intersection(hypothesis_tokens)
    false_positives = hypothesis_tokens.difference(reference_tokens)
    false_negatives = reference_tokens.difference(hypothesis_tokens)
    true_negatives = source_tokens.difference(reference_tokens).difference(hypothesis_tokens)

    return {"tp": list(true_positives), "fp": list(false_positives), "fn": list(false_negatives), "tn": list(true_negatives)}


if __name__ == '__main__':

    res = process_confusion_dict(
        source="Et elle se mit à pleurer de joie, et Benjamin aussi;",
        reference="FILLE PLEURER JOIE BENJAMIN AUSSI",
        hypothesis="JOIE METTRE PLEURER BENJAMIN AUSSI"
    )

    print("True Positives:", res["tp"])
    print("False Positives:", res["fp"])
    print("False Negatives:", res["fn"])
    print("True Negatives:", res["tn"])
