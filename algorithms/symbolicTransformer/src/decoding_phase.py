#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import torch
from docopt import docopt
import os

from common.constant import pretty_print_hypothesis, Tag, dir_separator
from common.output_decoder import greedy_decode, beam_search
from common.metrics.bleu import compute_corpus_level_bleu_score

from common.constant import EnvType
from algorithms.symbolicTransformer.src.core.architecture import NMT
from algorithms.symbolicTransformer.src.tools.attention_visualization import plot_attention_maps, get_decoder_self
from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.core.batching import create_dataloaders, Batch
from algorithms.symbolicTransformer.src.tools.helper import load_config


def check_outputs(
        model,
        vocab,
        dataloader_validation,
        n_examples=15):

    results = [()] * int(n_examples)
    for example_id in range(n_examples):

        print("\nExample %d ========\n" % example_id)

        # load a batch element
        data_val = next(iter(dataloader_validation))
        data_val_batch = Batch(data_val[0], data_val[1])

        # retrieve tokens with itos elements
        src_tokens = vocab.untokenize_src(data_val_batch.src[0])
        reference = vocab.untokenize_tgt(data_val_batch.tgt[0])

        # pretty print source and target
        vocab.pretty_print_token("Source Text (Input)        : ", src_tokens)
        vocab.pretty_print_token("Target Text (Ground Truth) : ", reference)

        # choose decoder style from config
        if int(learning_configuration["beam_search"]) == 1:

            hypothesis, estimation = beam_search(
                model,
                data_val_batch,
                learning_configuration,
                beam_size=int(learning_configuration["beam"]['beam-size']),
                max_decoding_time_step=int(learning_configuration["beam"]['max-decoding-time-step'])
            )

        else:

            hypothesis, estimation = greedy_decode(
                model,
                data_val_batch,
                learning_configuration["max_padding"]
            )

        # pretty print the model output
        pretty_print_hypothesis(hypothesis)

        model_output = (
                " ".join(
                    [vocab.tgt.get_itos()[x] for x in estimation[0] if x != Tag.BLANK.value[1]]
                ).split(str(Tag.STOP.value), 1)[0]
                + str(Tag.STOP.value)
        )
        print("Model Output               : " + model_output.replace("\n", ""))

        # run BLEU score
        results[example_id] = (data_val_batch, src_tokens, reference, estimation, model_output)

        reference = model.output_format_reference(reference)
        hypothesis = model.output_format_hypothesis(hypothesis)

        bleu_score = compute_corpus_level_bleu_score(reference, hypothesis)
        print(f"BLEU score * 100 : {bleu_score*100} ---")

    return results


def run_model_example(config, n_examples=5):

    vocab = Vocab(load_tokenizers(), config, EnvType.VALIDATION)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        vocab,
        torch.device("cpu"),
        target_mode=config["target_mode"],
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
    example_data = check_outputs(
        model,
        vocab,
        valid_dataloader,
        n_examples=n_examples)

    return model, example_data


# ---------------------------------------------------------------------------

if __name__ == '__main__':

    args = docopt(__doc__)
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator

    torch.cuda.empty_cache()

    learning_configuration = load_config()

    if not args['cpu']:
        learning_configuration["model_device"] = "cuda"
    else:
        learning_configuration["model_device"] = "cpu"

    model_learned, data_learned = run_model_example(config=learning_configuration)
    data_graph = data_learned[len(data_learned) - 1]

    plot_attention_maps(model_learned, data_learned, get_decoder_self, idx=0)

    # chart = visualize_layer(
    #     model_learned, 1, get_decoder_self, len(data_graph[1]), data_graph[1], data_graph[1]
    # )
    #
    # chart.save('output/translation_attention.html', embed_options={'renderer': 'svg'})
