from nltk.translate.bleu_score import corpus_bleu
from common.constant import Hypothesis, Tag

def output_format_reference(vec, output_max):
    delta = output_max - len(vec)
    res = vec + [str(Tag.BLANK.value[0])] * delta
    return [res]

def output_format_hypothesis(vec, output_max):
    delta = output_max - len(vec.value)
    res_val = vec.value + [str(Tag.BLANK.value[0])] * delta
    res = Hypothesis(value=res_val, score=vec.score)
    return [res]

def processing_bleu_score(reference, hypothesis, output_max=None, display=False, shrink=False):
    if shrink or (output_max is None):
        output_max = max(len(reference), len(hypothesis.value))

    bleu_score = corpus_bleu([
        [ref] for ref in output_format_reference(reference, output_max)],
        [hyp.value for hyp in output_format_hypothesis(hypothesis, output_max)])

    if display:
        print(f"BLEU score * 100 : {bleu_score*100} ---")

    return bleu_score
