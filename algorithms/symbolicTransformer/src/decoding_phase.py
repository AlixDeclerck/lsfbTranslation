#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import torch
from docopt import docopt
import os
from nltk.translate.bleu_score import corpus_bleu

from common.constant import pretty_print_hypothesis, Tag, dir_separator
from common.output_decoder import greedy_decode, beam_search
from common.metrics.bleu import compute_corpus_level_bleu_score

from common.constant import EnvType
from algorithms.symbolicTransformer.src.core.architecture import NMT
from algorithms.symbolicTransformer.src.functionnal.attention_visualization import plot_attention_maps, get_decoder_self
from algorithms.symbolicTransformer.src.functionnal.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.core.batching import create_dataloaders, Batch
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config


def check_outputs(
        model,
        vocab,
        dataloader_validation,
        n_examples=15):

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
        if learning_configuration["beam_search"]:

            hypothesis_beam, estimation_beam = beam_search(
                model,
                data_val_batch,
                beam_size=int(learning_configuration["beam"]['beam-size']),
                max_decoding_time_step=int(learning_configuration["beam"]['max-decoding-time-step'])
            )

            # pretty print the model output
            model_output_beam = pretty_print_hypothesis(hypothesis_beam, "beam")

            bleu_score = corpus_bleu([
                [ref] for ref in model.output_format_reference(reference)],
                [hyp.value for hyp in model.output_format_hypothesis(hypothesis_beam)])

            print(f"BLEU score * 100 : {bleu_score*100} ---")

        else:
            model_output_beam = None

        hypothesis_greedy, estimation_greedy = greedy_decode(
            model,
            data_val_batch,
            learning_configuration["max_padding"]
        )

        # pretty print the model output
        model_output_greedy = pretty_print_hypothesis(hypothesis_greedy, "greedy")

        bleu_score = corpus_bleu([
            [ref] for ref in model.output_format_reference(reference)],
            [hyp.value for hyp in model.output_format_hypothesis(hypothesis_greedy)])

        print(f"BLEU score * 100 : {bleu_score*100} ---")

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

    vocab = Vocab(load_tokenizers(), config)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        vocab,
        EnvType.TEST.value,
        torch.device("cpu"),
        architecture_dev_mode=config["architecture_dev_mode"],
        application_path=application_path,
        batch_size=1,
        is_distributed=False
    )

    print("Loading Trained Model ...")

    model = NMT(vocab, config)
    model.load_state_dict(
        torch.load(config["model_path"]+config["model_prefix"]+config["model_suffix"], map_location=torch.device("cpu"))
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
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator
    torch.cuda.empty_cache()
    learning_configuration = load_config()

    if not args['cpu']:
        learning_configuration["using_gpu"] = True
    else:
        learning_configuration["using_gpu"] = False

    # INFERENCE
    used_model, inferred_data = run_model_example(config=learning_configuration)

    # DISPLAY RESULT
    plot_attention_maps(used_model, inferred_data, get_decoder_self)
