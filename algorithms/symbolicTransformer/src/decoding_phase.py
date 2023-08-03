#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import os
from pathlib import Path

import pandas
import torch
from docopt import docopt

from algorithms.symbolicTransformer.src.core.architecture import NMT
from algorithms.symbolicTransformer.src.core.batching import Batch, collate_batch
from algorithms.symbolicTransformer.src.functionnal.data_preparation import retrieve_conte_dataset, Vocab
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import EnvType, Dialect, Corpus, HypothesisType, d_date, current_session
from common.metrics.bleu_score import Translation
from common.output_decoder import greedy_decode, beam_search


def run_inference(config, app, save_file):
    today = d_date()
    case = current_session()
    path = "../../../common/output/decoding_scores_"+today+"_"+str(case.value[1])+".csv"
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    hypothesis_beam = None
    estimation_beam = None
    limit = config["inference_decoding"]["max_number_of_inferences"]
    is_beam_search = config["inference_decoding"]["beam_search"]
    formated_test_dataset = []
    results = []
    df_scores = pandas.DataFrame({
        'title': None,
        'phrase': None,
        'precision': None,
        'bleu': None,
        'bp': None,
        'hypothesis_length': None,
        'reference_length': None,
        'unigram': None,
        'bigram': None,
        'trigram': None,
        'fourgram': None,
        'score_meteor': None,
        'detailed_meteor': None
    }, index=[0])

    # Load Vocabulary
    vocab = Vocab(config)
    vocab.retrieve_from_disk()
    print("Vocab loaded.. ")
    print("Source contains "+str(len(vocab.src))+" elements")
    print("Target contains "+str(len(vocab.tgt))+" elements")
    print()

    # Load test set
    test_dataset = retrieve_conte_dataset(
        EnvType.TEST.value,
        app,
        config["configuration_path"]["selected_db"],
        Dialect.LSF,
        vocab.is_english_output, False, 10000)
    print("Test set loaded.. ")
    print("With "+str(len(test_dataset))+" elements")
    print()

    # Instantiate NMT
    model = NMT(vocab, config)
    model.load_state_dict(
        torch.load(config["configuration_path"]["model_path"]+config["configuration_path"]["model_prefix"]+config["configuration_path"]["model_suffix"], map_location=torch.device("cpu"))
    )

    # Filter datas
    if bool(config["learning_config"]["english_output"]):
        filtered_test_dataset = pandas.DataFrame(test_dataset, columns=[Corpus.TEXT_FR.value[2], Corpus.TEXT_EN.value[2], Corpus.GLOSS_LSF.value[2]])[[Corpus.TEXT_FR.value[2], Corpus.TEXT_EN.value[2]]].to_numpy()
    else:
        filtered_test_dataset = pandas.DataFrame(test_dataset, columns=[Corpus.TEXT_FR.value[2], Corpus.TEXT_EN.value[2], Corpus.GLOSS_LSF.value[2]])[[Corpus.TEXT_FR.value[2], Corpus.GLOSS_LSF.value[2]]].to_numpy()

    for f in filtered_test_dataset:
        lst = [f]
        formated_test_dataset.append(lst)

    # Inference
    for i, data in enumerate(formated_test_dataset):
        trans = Translation(config, data[0][0], data[0][1])
        data_val = collate_batch(data, vocab, model.device)
        data_val_batch = Batch(data_val[0], data_val[1])

        # DECODING
        if is_beam_search:
            beam_search(model, data_val_batch, beam_size=5, max_decoding_time_step=24)
            hypothesis_beam, estimation_beam = beam_search(
                model,
                data_val_batch,
                beam_size=int(config["inference_decoding"]['beam-size']),
                max_decoding_time_step=int(config["inference_decoding"]['max-decoding-time-step'])
            )

        greedy_decode(model, data_val_batch, config["learning_config"]["max_padding"])
        hypothesis_greedy, estimation_greedy = greedy_decode(
            model,
            data_val_batch,
            config["learning_config"]["max_padding"]
        )

        # SCORING
        title = str(i+1)+". Traduction de : "
        if is_beam_search:
            trans.add_hypothesis(HypothesisType.BEAM, hypothesis_beam)
        trans.add_hypothesis(HypothesisType.GREEDY, hypothesis_greedy)
        trans.display_translation(title)
        trans.export(title, df_scores)

        # CONSTRUCT RESULT VALUE LIST
        results.append([
            data_val_batch,
            trans,
            estimation_greedy,
            estimation_beam])

        if not save_file:
            return model, results

        elif i+1 == limit:
            df_scores.to_csv(filepath)
            return model, results

    if save_file:
        df_scores.to_csv(filepath)

    return model, results


# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # CONFIGURATION
    args = docopt(__doc__)
    learning_configuration = load_config()
    application_path = os.environ['HOME'] + learning_configuration["configuration_path"]["application_path"] + args['--app-path'] + learning_configuration["configuration_path"]["application_path"]
    torch.cuda.empty_cache()

    if not args['cpu']:
        learning_configuration["learning_config"]["using_gpu"] = True
    else:
        learning_configuration["learning_config"]["using_gpu"] = False

    # INFERENCE
    used_model, inferred_data = run_inference(config=learning_configuration, app=application_path, save_file=True)
    # plot_attention_maps(used_model, inferred_data, learning_configuration)
