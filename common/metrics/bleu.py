"""
Original source :
CS224N 2019-20: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

from typing import List
from nltk.translate.bleu_score import corpus_bleu
from common.constant import Hypothesis, Tag


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]):  # -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # if references[0][0] == '<s>':
    #     references = [ref[1:-1] for ref in references]
    #
    # length_references = len(references)
    # length_hypothesis = len(hypotheses)
    # delta = length_references - length_hypothesis
    #
    # if delta > 0:
    #     for d in range(delta):
    #         hypotheses.append(Hypothesis(value=str(Tag.UNKNOWN), score=0))
    # elif delta < 0:
    #     for d in range(abs(delta)):
    #         references.append([str(Tag.UNKNOWN)])

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score
