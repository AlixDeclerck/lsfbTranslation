import evaluate
import spacy
from algorithms.syntax_analysis.with_spacy.phrases import SpacyPhrase
from common.constant import HypothesisType, Tag, Hypothesis
from common.metrics import pymeteor, roc

class Translation:

    def __init__(self, config, source_text, reference):
        # self.nlp = spacy.load("fr_core_news_sm")
        self.bleu_eval = evaluate.load("bleu")
        self.N = config["inference_decoding"]['output_max_words']
        self.source_text = source_text
        self.reference = [[reference]]
        self.approximated_hypothesis = self.update_approx()
        self.approximated_hypothesis_meteor = pymeteor.meteor(self.reference[0][0], self.approximated_hypothesis["text"][0], detailed_result=True)
        self.approximated_roc = roc.process_confusion_dict(source=source_text, reference=self.reference[0][0], hypothesis=self.approximated_hypothesis["text"][0])
        self.beam_hypothesis = None
        self.beam_score = None
        self.beam_meteor = None
        self.beam_roc = None
        self.greedy_hypothesis = None
        self.greedy_score = None
        self.greedy_meteor = None
        self.greedy_roc = None

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
            self.beam_meteor = pymeteor.meteor(self.reference[0][0], self.beam_hypothesis, detailed_result=True)
            self.beam_roc = roc.process_confusion_dict(source=self.source_text, reference=self.reference[0][0], hypothesis=self.beam_hypothesis)
        elif HypothesisType.GREEDY == hypothesis_type:
            self.greedy_score = result
            self.greedy_hypothesis = hs
            self.greedy_meteor = pymeteor.meteor(self.reference[0][0], self.greedy_hypothesis, detailed_result=True)
            self.greedy_roc = roc.process_confusion_dict(source=self.source_text, reference=self.reference[0][0], hypothesis=self.greedy_hypothesis)
        else:
            return result

    def update_approx(self):
        approx = self.approximation(self.source_text)
        output = Tag.START.value[0]+" "+approx+" "+Tag.STOP.value[0]
        hyp = Hypothesis(value=output.split(" "), score=0.0)
        return self.add_hypothesis(HypothesisType.APPROX, hyp)

    def process_bleu_score(self, hypothesis):
        return self.bleu_eval.compute(references=self.reference, predictions=hypothesis)

    def process_precision_n(self, n, hypothesis):
        bleu_score = self.process_bleu_score([hypothesis])
        p1 = bleu_score["precisions"][0]
        p2 = bleu_score["precisions"][1]
        p3 = bleu_score["precisions"][2]
        p4 = bleu_score["precisions"][3]

        if n == 1:
            res = p1
        elif n == 2:
            res = (p1 * p2)**(1/2)
        elif n == 3:
            res = (p1 * p2 * p3)**(1/3)
        else:
            res = (p1 * p2 * p3 * p4)**(1/4)

        return round(res, 2)

    @staticmethod
    def approximation(src_txt):
        nlp = spacy.load("fr_core_news_sm")
        phrases = SpacyPhrase(nlp(src_txt))
        phrases.preprocessing()
        phrases.handle_scenes()
        phrases.grammar_handler()
        return phrases.handle_output(glosses=True).split("|")[0]

    @staticmethod
    def print_line(txt, score, unigram, bigram, trigram, fourgram, meteor, detailed_meteor, roc_confusion):
        print(txt + "| precision : " + str(score["precisions"]) + " | score BLEU : " + str(round(score["bleu"], 2)) + " | BP : " + str(round(score["brevity_penalty"], 2)) + " | unigram : " + str(unigram) + " | bigram : " + str(bigram) + " | trigram : " + str(trigram) + " | 4gram : " + str(fourgram) + " | meteor : " + str(meteor) + " | detailed_meteor : " + str(detailed_meteor) + " | roc confusion : " + str(roc_confusion))

    def display_translation(self, title):
        print(title+str(self.source_text)+"| reference : "+str(self.reference[0][0]))
        self.print_line(
            txt="Approximation | "+str(self.approximated_hypothesis["text"][0]),
            score=self.approximated_hypothesis,
            unigram=self.process_precision_n(1, self.approximated_hypothesis["text"][0]),
            bigram=self.process_precision_n(2, self.approximated_hypothesis["text"][0]),
            trigram=self.process_precision_n(3, self.approximated_hypothesis["text"][0]),
            fourgram=self.process_precision_n(4, self.approximated_hypothesis["text"][0]),
            meteor=self.approximated_hypothesis_meteor["score"],
            detailed_meteor=self.approximated_hypothesis_meteor,
            roc_confusion=self.approximated_roc
        )

        if self.beam_score is not None:
            self.print_line(
                txt="Beam | " + str(self.beam_hypothesis),
                score=self.beam_score,
                unigram=self.process_precision_n(1, self.beam_hypothesis),
                bigram=self.process_precision_n(2, self.beam_hypothesis),
                trigram=self.process_precision_n(3, self.beam_hypothesis),
                fourgram=self.process_precision_n(4, self.beam_hypothesis),
                meteor=self.beam_meteor["score"],
                detailed_meteor=self.beam_meteor,
                roc_confusion=self.beam_roc
            )

        if self.greedy_score is not None:
            self.print_line(
                txt="Greedy | " + str(self.greedy_hypothesis),
                score=self.greedy_score,
                unigram=self.process_precision_n(1, self.greedy_hypothesis),
                bigram=self.process_precision_n(2, self.greedy_hypothesis),
                trigram=self.process_precision_n(3, self.greedy_hypothesis),
                fourgram=self.process_precision_n(4, self.greedy_hypothesis),
                meteor=self.greedy_meteor["score"],
                detailed_meteor=self.greedy_meteor,
                roc_confusion=self.greedy_roc
            )

        print("\n")

    def export(self, title, dataframe):
        dataframe.loc[len(dataframe.index)] = [
            title, self.source_text, self.reference[0][0],
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        ]
        dataframe.loc[len(dataframe.index)] = [
            "Approximation",
            str(self.approximated_hypothesis["text"][0]),
            self.approximated_hypothesis["precisions"],
            round(self.approximated_hypothesis["bleu"], 2),
            round(self.approximated_hypothesis["brevity_penalty"], 2),
            self.approximated_hypothesis["translation_length"],
            self.approximated_hypothesis["reference_length"],
            self.process_precision_n(1, self.approximated_hypothesis["text"][0]),
            self.process_precision_n(2, self.approximated_hypothesis["text"][0]),
            self.process_precision_n(3, self.approximated_hypothesis["text"][0]),
            self.process_precision_n(4, self.approximated_hypothesis["text"][0]),
            self.approximated_hypothesis_meteor["score"],
            self.approximated_hypothesis_meteor,
            self.approximated_roc["tp"],
            self.approximated_roc["fp"],
            self.approximated_roc["tn"],
            self.approximated_roc["fn"],
            self.source_text,
            self.reference[0][0],
            self.approximated_hypothesis["text"][0]
        ]

        if self.beam_score is not None:
            dataframe.loc[len(dataframe.index)] = [
                "Beam",
                str(self.beam_score["text"][0]),
                self.beam_score["precisions"],
                round(self.beam_score["bleu"], 2),
                round(self.beam_score["brevity_penalty"], 2),
                self.beam_score["translation_length"],
                self.beam_score["reference_length"],
                self.process_precision_n(1, self.beam_hypothesis),
                self.process_precision_n(2, self.beam_hypothesis),
                self.process_precision_n(3, self.beam_hypothesis),
                self.process_precision_n(4, self.beam_hypothesis),
                self.beam_meteor["score"],
                self.beam_meteor,
                self.beam_roc["tp"],
                self.beam_roc["fp"],
                self.beam_roc["tn"],
                self.beam_roc["fn"],
                self.source_text,
                self.reference[0][0],
                self.beam_hypothesis
            ]

        dataframe.loc[len(dataframe.index)] = [
            "Greedy",
            str(self.greedy_score["text"][0]),
            self.greedy_score["precisions"],
            round(self.greedy_score["bleu"], 2),
            round(self.greedy_score["brevity_penalty"], 2),
            self.greedy_score["translation_length"],
            self.greedy_score["reference_length"],
            self.process_precision_n(1, self.greedy_hypothesis),
            self.process_precision_n(2, self.greedy_hypothesis),
            self.process_precision_n(3, self.greedy_hypothesis),
            self.process_precision_n(4, self.greedy_hypothesis),
            self.greedy_meteor["score"],
            self.greedy_meteor,
            self.greedy_roc["tp"],
            self.greedy_roc["fp"],
            self.greedy_roc["tn"],
            self.greedy_roc["fn"],
            self.source_text,
            self.reference[0][0],
            self.greedy_hypothesis
        ]
        return dataframe
