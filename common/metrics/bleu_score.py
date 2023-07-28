import evaluate
import spacy
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from algorithms.syntax_analysis.with_spacy.phrases import SpacyPhrase
from common.constant import HypothesisType, Tag, Hypothesis
from nltk.translate.bleu_score import sentence_bleu

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

    def process_bleu_n_score(self, n, hypothesis):
        if n == 1:
            w = (1, 0, 0, 0)
        elif n == 2:
            w = (0.5, 0.5, 0, 0)
        elif n == 3:
            w = (0.33, 0.33, 0.33, 0)
        else:
            w = (0.25, 0.25, 0.25, 0.25)

        return round(sentence_bleu(self.reference, hypothesis, weights=w), 2)

    @staticmethod
    def print_line(txt, score, unigram, bigram, trigram, fourgram):
        print(txt + "| precision : " + str(score["precisions"]) + " | score BLEU : " + str(round(score["bleu"], 2)) + " | BP : " + str(round(score["brevity_penalty"], 2)) + " | unigram : " + str(unigram) + " | bigram : " + str(bigram) + " | trigram : " + str(trigram) + " | 4gram : " + str(fourgram))

    def display_translation(self, title):
        print(title+str(self.source_text)+"| reference : "+str(self.reference[0][0]))
        self.print_line(
            txt="Approximation | "+str(self.approximated_hypothesis["text"][0]),
            score=self.approximated_hypothesis,
            unigram=self.process_bleu_n_score(1, self.approximated_hypothesis["text"][0]),
            bigram=self.process_bleu_n_score(2, self.approximated_hypothesis["text"][0]),
            trigram=self.process_bleu_n_score(3, self.approximated_hypothesis["text"][0]),
            fourgram=self.process_bleu_n_score(4, self.approximated_hypothesis["text"][0])
        )

        if self.beam_score is not None:
            self.print_line(
                txt="Beam | " + str(self.beam_hypothesis),
                score=self.beam_score,
                unigram=self.process_bleu_n_score(1, self.beam_hypothesis),
                bigram=self.process_bleu_n_score(2, self.beam_hypothesis),
                trigram=self.process_bleu_n_score(3, self.beam_hypothesis),
                fourgram=self.process_bleu_n_score(4, self.beam_hypothesis)
            )

        if self.greedy_score is not None:
            self.print_line(
                txt="Greedy | " + str(self.greedy_hypothesis),
                score=self.greedy_score,
                unigram=self.process_bleu_n_score(1, self.greedy_hypothesis),
                bigram=self.process_bleu_n_score(2, self.greedy_hypothesis),
                trigram=self.process_bleu_n_score(3, self.greedy_hypothesis),
                fourgram=self.process_bleu_n_score(4, self.greedy_hypothesis)
            )

        print("\n")

    def export(self, title, dataframe):
        dataframe.loc[len(dataframe.index)] = [
            title,
            self.source_text,
            self.reference[0][0],
            None,
            None,
            None,
            None,
            None,
            None
        ]
        dataframe.loc[len(dataframe.index)] = [
            "Approximation",
            str(self.approximated_hypothesis["text"][0]),
            self.approximated_hypothesis["precisions"],
            round(self.approximated_hypothesis["bleu"], 2),
            round(self.approximated_hypothesis["brevity_penalty"], 2),
            self.process_bleu_n_score(1, self.approximated_hypothesis["text"][0]),
            self.process_bleu_n_score(2, self.approximated_hypothesis["text"][0]),
            self.process_bleu_n_score(3, self.approximated_hypothesis["text"][0]),
            self.process_bleu_n_score(4, self.approximated_hypothesis["text"][0])
        ]

        if self.beam_score is not None:
            dataframe.loc[len(dataframe.index)] = [
                "Beam",
                str(self.beam_score["text"][0]),
                self.beam_score["precisions"],
                round(self.beam_score["bleu"], 2),
                round(self.beam_score["brevity_penalty"], 2),
                self.process_bleu_n_score(1, self.beam_hypothesis),
                self.process_bleu_n_score(2, self.beam_hypothesis),
                self.process_bleu_n_score(3, self.beam_hypothesis),
                self.process_bleu_n_score(4, self.beam_hypothesis)
            ]

        dataframe.loc[len(dataframe.index)] = [
            "Greedy",
            str(self.greedy_score["text"][0]),
            self.greedy_score["precisions"],
            round(self.greedy_score["bleu"], 2),
            round(self.greedy_score["brevity_penalty"], 2),
            self.process_bleu_n_score(1, self.greedy_hypothesis),
            self.process_bleu_n_score(2, self.greedy_hypothesis),
            self.process_bleu_n_score(3, self.greedy_hypothesis),
            self.process_bleu_n_score(4, self.greedy_hypothesis)
        ]
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
