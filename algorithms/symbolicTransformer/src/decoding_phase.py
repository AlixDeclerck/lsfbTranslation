#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

from pathlib import Path
import os
import pandas
import torch
from docopt import docopt
from algorithms.symbolicTransformer.src.core.architecture import NMT
from algorithms.symbolicTransformer.src.core.batching import Batch, collate_batch
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from algorithms.symbolicTransformer.src.functionnal.data_preparation import retrieve_conte_dataset, Vocab
from algorithms.symbolicTransformer.src.functionnal.attention_visualization import plot_attention_maps
from common.constant import EnvType, Dialect, Corpus, HypothesisType
from common.output_decoder import greedy_decode, beam_search
from common.metrics.bleu_score import Translation


def run_inference(config):

    filepath = Path('../../../common/output/decoding_scores_2023-07-22.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    hypothesis_beam = None
    limit = config["inference_decoding"]["max_number_of_inferences"]
    is_beam_search = learning_configuration["inference_decoding"]["beam_search"]
    formated_test_dataset = []
    results = []
    df_scores = pandas.DataFrame({
        'title': None,
        'phrase': None,
        'precision': None,
        'bleu': None,
        'bp': None,
        'trigram': None
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
        application_path,
        config["configuration_path"]["selected_db"],
        Dialect.LSF,
        vocab.english_output, False, 10000)
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
            hypothesis_beam, estimation_beam = beam_search(
                model,
                data_val_batch,
                beam_size=int(learning_configuration["inference_decoding"]['beam-size']),
                max_decoding_time_step=int(learning_configuration["inference_decoding"]['max-decoding-time-step'])
            )

        hypothesis_greedy, estimation_greedy = greedy_decode(
            model,
            data_val_batch,
            learning_configuration["learning_config"]["max_padding"]
        )

        # SCORING
        title = str(i+1)+". Traduction de : "
        trans.add_hypothesis(HypothesisType.BEAM, hypothesis_beam)
        trans.add_hypothesis(HypothesisType.GREEDY, hypothesis_greedy)
        trans.display_translation(title)
        trans.export(title, df_scores)

        # CONSTRUCT RESULT VALUE LIST
        results.append([
            data_val_batch,
            trans.source_text,
            trans.reference,
            estimation_greedy,
            trans.beam_hypothesis,
            trans.greedy_hypothesis])

        if i == limit:
            df_scores.to_csv(filepath)
            return model, results

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
    used_model, inferred_data = run_inference(config=learning_configuration)
    # plot_attention_maps(used_model, inferred_data, learning_configuration)
