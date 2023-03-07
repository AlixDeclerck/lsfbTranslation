#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    with_spacy.py train --app-path=<file>
"""

import os
import spacy
import textacy
from collections import Counter
from docopt import docopt

from algorithms.data_loader.src.dal import EnvType
from algorithms.symbolicTransformer.src.core.data_preparation import retrieve_phoenix_dataset

def introduction(txt):
    print("file from type "+str(type(txt)))
    print([token.text for token in nlp(txt)])


# Sentence detection is the process of locating where sentences start and end in a given text.
def sentences(txt):
    about_doc = nlp(txt)
    sent = list(about_doc.sents)
    print("sentences a len : ", str(len(sent)))

    for s in sent:
        print(f"{s[:5]}...")


# The process of tokenization breaks a text down into its basic units—or tokens—which are represented in spaCy as Token objects.
def tokenization_index(txt):
    about_doc = nlp(txt)

    for token in about_doc:
        print(token, token.idx)


# Examples of attributes
# https://spacy.io/api/token#attributes
def attributes(txt):
    print("\n")
    print(
        f"{'Text with Whitespace':22}"
        f"{'Is Alphanumeric?':15}"
        f"{'Is Punctuation?':18}"
        f"{'Is Stop Word?'}"
    )

    for token in nlp(txt):
        print(
            f"{str(token.text_with_ws):22}"
            f"{str(token.is_alpha):15}"
            f"{str(token.is_punct):18}"
            f"{str(token.is_stop)}"
        )


# typically stop words are removed in NLP
def remove_stop_words(txt):
    about_doc = nlp(txt)
    print([token for token in about_doc if not token.is_stop])


# Lemmatization is the process of reducing inflected forms of a word while still ensuring that the reduced form belongs to the language. This reduced form, or root word, is called a lemma.
def lemmatization(txt):
    conference_help_doc = nlp(txt)
    for token in conference_help_doc:
        if str(token) != str(token.lemma_):
            print(f"{str(token):>20} : {str(token.lemma_)}")


# word_frequency analysis
def word_frequency(txt):
    complete_doc = nlp(txt)

    words = [
        token.text
        for token in complete_doc
        if not token.is_stop and not token.is_punct
    ]

    print(Counter(words).most_common(5))


# word_frequency analysis
def words_frequencies(learning_corpus):
    words = learning_corpus[0][0]
    for i in range(1, 20):
        words += learning_corpus[i][0]

    word_frequency(words)


# Part of speech or POS is a grammatical role that explains how a particular word is used in a sentence. There are typically eight parts of speech:
# Part-of-speech tagging is the process of assigning a POS tag to each token depending on its usage in the sentence. POS tags are useful for assigning a syntactic category like noun or verb to each word.
def part_of_speech(txt):
    about_doc = nlp(txt)
    for token in about_doc:
        print(
            f"""
    TOKEN: {str(token)}
    =====
    TAG: {str(token.tag_):10} POS: {token.pos_}
    EXPLANATION: {spacy.explain(token.tag_)}"""
        )


def pos_categories(txt):
    nouns = []
    adjectives = []
    about_doc = nlp(txt)

    for token in about_doc:
        if token.pos_ == "NOUN":
            nouns.append(token)
        if token.pos_ == "ADJ":
            adjectives.append(token)

    print(f"nouns : {nouns}")
    print(f"adjectives : {adjectives}")


def preprocessing(txt):
    complete_doc = nlp(txt)

    def is_token_allowed(token):
        return bool(
            token
            and str(token).strip()
            and not token.is_stop
            and not token.is_punct
        )

    def preprocess_token(token):
        return token.lemma_.strip().lower()

    complete_filtered_tokens = [
        preprocess_token(token)
        for token in complete_doc
        if is_token_allowed(token)
    ]

    print(complete_filtered_tokens)


def dependency_parsing(txt):
    piano_doc = nlp(txt)
    for token in piano_doc:
        print(
            f"""
    TOKEN: {token.text}
    =====
    {token.tag_ = }
    {token.head.text = }
    {token.dep_ = }"""
        )


def shallow_parsing_noun(txt):
    conference_doc = nlp(txt)

    # Extract Noun Phrases
    for chunk in conference_doc.noun_chunks:
        print (chunk)


def shallow_parsing_verb(txt):
    patterns = [{"POS": "AUX"}, {"POS": "VERB"}]
    about_talk_doc = textacy.make_spacy_doc(
        txt, lang="en_core_web_sm"
    )
    verb_phrases = textacy.extract.token_matches(
        about_talk_doc, patterns=patterns
    )

    print("Print all verb phrases")
    for chunk in verb_phrases:
        print(chunk.text)

    print("\nExtract noun phrase to explain what nouns are involved")
    for chunk in about_talk_doc.noun_chunks:
        print(chunk)


def named_entity_recognition(txt):
    piano_class_text = (
        "Great Piano Academy is situated"
        " in Mayfair or the City of London and has"
        " world-class piano instructors."
    )
    piano_class_doc = nlp(piano_class_text)

    for ent in piano_class_doc.ents:
        print(
            f"""
    {ent.text = }
    {ent.start_char = }
    {ent.end_char = }
    {ent.label_ = }
    spacy.explain('{ent.label_}') = {spacy.explain(ent.label_)}"""
        )


def named_entity_recognition_survey(txt):
    survey_text = (
        "Out of 5 people surveyed, James Robert,"
        " Julie Fuller and Benjamin Brooks like"
        " apples. Kelly Cox and Matthew Evans"
        " like oranges."
    )

    def replace_person_names(token):
        if token.ent_iob != 0 and token.ent_type_ == "PERSON":
            return "[REDACTED] "
        return token.text_with_ws

    def redact_names(nlp_doc):
        with nlp_doc.retokenize() as retokenizer:
            for ent in nlp_doc.ents:
                retokenizer.merge(ent)
        tokens = map(replace_person_names, nlp_doc)
        return "".join(tokens)

    survey_doc = nlp(survey_text)
    print(redact_names(survey_doc))


# ---------------------------------------------------------------------------------------


def pied():
    learning_corpus = retrieve_phoenix_dataset(EnvType.DEV, application_path)

    introduction(learning_corpus[0][0])
    sentences(learning_corpus[0][0])
    tokenization_index(learning_corpus[0][0])
    attributes(learning_corpus[0][0])
    words_frequencies(learning_corpus)
    named_entity_recognition(learning_corpus[0][0])
    named_entity_recognition_survey(learning_corpus[0][0])


def main():
    learning_corpus = retrieve_phoenix_dataset(EnvType.DEV, application_path)

    # remove_stop_words(learning_corpus[0][0])
    # lemmatization(learning_corpus[0][0])
    # part_of_speech(learning_corpus[0][0])
    # pos_categories(learning_corpus[0][0])
    preprocessing(learning_corpus[0][0])
    # dependency_parsing(learning_corpus[0][0])
    # shallow_parsing_noun(learning_corpus[0][0])
    # shallow_parsing_verb(learning_corpus[0][0])


# ---------------------------------------------------------------------------------------

global nlp, application_path

dir_separator = "/"                     # linux folder structure
nlp = spacy.load("fr_core_news_sm")     # nlp : the doc object
args = docopt(__doc__)                  # read application parameters

# Construct application path
application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

if __name__ == '__main__':
    main()
