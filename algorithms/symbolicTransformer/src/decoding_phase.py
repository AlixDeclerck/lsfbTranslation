#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import os
import torch
from docopt import docopt

from algorithms.symbolicTransformer.src.core.architecture import NMT
from algorithms.symbolicTransformer.src.core.batching import create_dataloaders, Batch
from algorithms.symbolicTransformer.src.functionnal.attention_visualization import plot_attention_maps, get_decoder_self
from algorithms.symbolicTransformer.src.functionnal.data_preparation import Vocab
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import EnvType
from common.constant import pretty_print_hypothesis
from common.metrics.bleu import processing_bleu_score
from common.output_decoder import greedy_decode, beam_search

"""
The decoding phase contents are coming from :
Annotated transformer
Huang, et al. 2022 / Rush, et al. 2019
nlp.seas.harvard.edu/annotated-transformer
"""

def check_outputs(model, vocab, dataloader_validation, n_examples=15):

    results = []
    for example_id in range(n_examples):

        print("\nExample %d ========\n" % example_id)

        # load a batch element
        data_val = next(iter(dataloader_validation))
        data_val_batch = Batch(data_val[0], data_val[1])

        # retrieve tokens with itos elements
        src_tokens = vocab.unembed_src(data_val_batch.src[0])
        reference = vocab.unembed_tgt(data_val_batch.tgt[0])

        # pretty print source and target
        vocab.pretty_print_token("Source Text (Input)        : ", src_tokens)
        vocab.pretty_print_token("Target Text (Ground Truth) : ", reference)

        # DECODING
        if not learning_configuration["inference_decoding"]["beam_search"]:
            model_output_beam = None

        else:
            hypothesis_beam, estimation_beam = beam_search(
                model,
                data_val_batch,
                beam_size=int(learning_configuration["inference_decoding"]['beam-size']),
                max_decoding_time_step=int(learning_configuration["inference_decoding"]['max-decoding-time-step'])
            )

            # pretty print the model output
            model_output_beam = pretty_print_hypothesis(hypothesis_beam, "beam")
            processing_bleu_score(reference,  hypothesis_beam, output_max=learning_configuration["learning_config"]["output_max_words"], display=True)

        hypothesis_greedy, estimation_greedy = greedy_decode(
            model,
            data_val_batch,
            learning_configuration["learning_config"]["max_padding"]
        )

        # pretty print the model output
        model_output_greedy = pretty_print_hypothesis(hypothesis_greedy, "greedy")
        processing_bleu_score(reference,  hypothesis_greedy, output_max=learning_configuration["learning_config"]["output_max_words"], display=True)

        # CONSTRUCT RESULT VALUE LIST
        results.append([
            data_val_batch,
            src_tokens,
            reference,
            estimation_greedy,
            model_output_beam,
            model_output_greedy])

    return results


def run_model_example(config, n_examples=5):

    vocab = Vocab(config)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        vocab,
        EnvType.TEST.value,
        torch.device("cpu"),
        english_output=config["learning_config"]["english_output"],
        application_path=application_path,
        selected_db=config["configuration_path"]["selected_db"],
        batch_size=1,
        is_distributed=False
    )

    print("Loading Trained Model ...")

    model = NMT(vocab, config)
    model.load_state_dict(
        torch.load(config["configuration_path"]["model_path"]+config["configuration_path"]["model_prefix"]+config["configuration_path"]["model_suffix"], map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    inferred_outputs = check_outputs(
        model,
        vocab,
        valid_dataloader,
        n_examples=n_examples)

    return model, inferred_outputs


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
    used_model, inferred_data = run_model_example(config=learning_configuration, n_examples=18)

    # DISPLAY RESULT
    plot_attention_maps(used_model, inferred_data, get_decoder_self)
