import evaluate
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from algorithms.syntax_analysis.with_spacy.phrases import SpacyPhrase
import spacy

class Translation:

    def __init__(self, source_text, reference, beam_hypothesis, greedy_hypothesis, config):
        self.source_text = source_text
        self.reference = [[reference]]
        self.beam_hypothesis = [beam_hypothesis]
        self.greedy_hypothesis = [greedy_hypothesis]
        self.approximated_hypothesis = self.approximate()
        self.N = config["learning_config"]['output_max_words']
        self.bleu_eval = evaluate.load("bleu")

    def approximate(self):
        nlp = spacy.load("fr_core_news_sm")
        phrases = SpacyPhrase(nlp(self.source_text))
        phrases.preprocessing()
        phrases.handle_scenes()
        phrases.grammar_handler()
        return phrases.handle_output(glosses=True).split("|")[0]

    def process_bleu_score(self):
        return self.bleu_eval.compute(references=self.reference, predictions=self.beam_hypothesis)


if __name__ == '__main__':

    cfg = load_config("../../algorithms/symbolicTransformer/src/config.yaml")

    # translation = Translation(
    #     source_text="Méfie - toi de ta belle-mère !",
    #     reference="ATTENTION TA BELLE MERE",
    #     beam_hypothesis="EH !",
    #     greedy_hypothesis="TOI BELLE MERE",
    #     config=cfg
    # )
    #
    # print("BLEU score : ", translation.process_bleu_score())
    #
    # translation = Translation(
    #     source_text="le cheval s' élancera dans les airs avec lui ,",
    #     reference="CHEVAL S ' ELANCER PARTIR DANS AIR AVEC ROI ",
    #     beam_hypothesis="CHEVAL CRIER",
    #     greedy_hypothesis="CHEVAL MONTER DANS FORET",
    #     config=cfg
    # )
    #
    # print("BLEU score : ", translation.process_bleu_score())

    translation = Translation(
        source_text="Le roi se réjouit de la trouver innocente ,",
        reference="ROI CONTENT COMPRENDRE SAVOIR ELLE INNOCENTE",
        beam_hypothesis="ROI CONTENT",
        greedy_hypothesis="ROI CONTENT ELLE",
        config=cfg
    )

    translation.approximate()
    print("BLEU score : ", translation.process_bleu_score())
