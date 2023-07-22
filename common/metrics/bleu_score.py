import evaluate
import spacy
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from algorithms.syntax_analysis.with_spacy.phrases import SpacyPhrase
from common.constant import HypothesisType, Tag, Hypothesis

class Translation:

    def __init__(self, config, source_text, reference):
        self.bleu_eval = evaluate.load("bleu")
        self.N = config["learning_config"]['output_max_words']
        self.source_text = source_text
        self.reference = [[reference]]
        self.approximated_hypothesis = self.approximate()
        self.beam_hypothesis = None
        self.beam_score = None
        self.greedy_hypothesis = None
        self.greedy_score = None

    def add_hypothesis(self, hypothesis_type, hypothesis):
        h = hypothesis[0][1:-1]
        hs = ""
        for s in h:
            hs += s+" "

        result = self.process_bleu_score([hs])
        result["text"] = [hs]
        if HypothesisType.BEAM == hypothesis_type:
            self.beam_score = result
            self.beam_hypothesis = hs
        elif HypothesisType.GREEDY == hypothesis_type:
            self.greedy_score = result
            self.greedy_hypothesis = hs
        else:
            return result

    def approximate(self):
        nlp = spacy.load("fr_core_news_sm")
        phrases = SpacyPhrase(nlp(self.source_text))
        phrases.preprocessing()
        phrases.handle_scenes()
        phrases.grammar_handler()
        output = Tag.START.value[0]+" "+phrases.handle_output(glosses=True).split("|")[0]+" "+Tag.STOP.value[0]
        hyp = Hypothesis(value=output.split(" "), score=0.0)
        return self.add_hypothesis(HypothesisType.APPROX, hyp)

    def process_bleu_score(self, hypothesis):
        return self.bleu_eval.compute(references=self.reference, predictions=hypothesis)

    @staticmethod
    def print_line(txt, score):
        trigram_score = round((score["precisions"][0]+score["precisions"][1]+score["precisions"][2]) / 3, 2)
        print(txt + "| precision : " + str(score["precisions"]) + " | score BLEU : " + str(round(score["bleu"], 2)) + " | BP : " + str(round(score["brevity_penalty"], 2)) + " | trigram : " + str(trigram_score))

    def display_translation(self, title):
        print(title+str(self.source_text)+"| reference : "+str(self.reference[0][0]))
        self.print_line("Approximation | "+str(self.approximated_hypothesis["text"][0]), self.approximated_hypothesis)

        if self.beam_score is not None:
            self.print_line("Beam | " + str(self.beam_hypothesis), self.beam_score)

        if self.greedy_score is not None:
            self.print_line("Greedy | " + str(self.greedy_hypothesis), self.greedy_score)

        print("\n")

    def export(self, title, dataframe):
        dataframe.loc[len(dataframe.index)] = [title, self.source_text, self.reference[0][0], None, None, None]
        dataframe.loc[len(dataframe.index)] = ["Approximation", str(self.approximated_hypothesis["text"][0]), self.approximated_hypothesis["precisions"], round(self.approximated_hypothesis["bleu"], 2), round(self.approximated_hypothesis["brevity_penalty"], 2), round((self.approximated_hypothesis["precisions"][0]+self.approximated_hypothesis["precisions"][1]+self.approximated_hypothesis["precisions"][2]) / 3, 2)]
        dataframe.loc[len(dataframe.index)] = ["Beam", str(self.beam_score["text"][0]), self.beam_score["precisions"], round(self.beam_score["bleu"], 2), round(self.beam_score["brevity_penalty"], 2), round((self.beam_score["precisions"][0]+self.beam_score["precisions"][1]+self.beam_score["precisions"][2]) / 3, 2)]
        dataframe.loc[len(dataframe.index)] = ["Greedy", str(self.greedy_score["text"][0]), self.greedy_score["precisions"], round(self.greedy_score["bleu"], 2), round(self.greedy_score["brevity_penalty"], 2), round((self.greedy_score["precisions"][0]+self.greedy_score["precisions"][1]+self.greedy_score["precisions"][2]) / 3, 2)]
        return dataframe


if __name__ == '__main__':

    cfg = load_config("../../algorithms/symbolicTransformer/src/config.yaml")

    translation = Translation(
        config=cfg,
        source_text="Le roi se réjouit de la trouver innocente ,",
        reference="ROI CONTENT COMPRENDRE SAVOIR ELLE INNOCENTE"
    )

    translation.add_hypothesis(HypothesisType.BEAM, Hypothesis(value=str(Tag.START.value[0]+" "+"ROI CONTENT"+" "+Tag.STOP.value[0]).split(" "), score=0.0))
    translation.add_hypothesis(HypothesisType.GREEDY, Hypothesis(value=str(Tag.START.value[0]+" "+"ROI CONTENT ELLE"+" "+Tag.STOP.value[0]).split(" "), score=0.0))
    translation.display_translation("Traduction de ")
