#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import torch
from common.constant import dir_separator
from docopt import docopt
import os

from common.constant import Tag
from algorithms.data_loader.src.dal import EnvType
from algorithms.symbolicTransformer.src.core.architecture import make_model
from algorithms.symbolicTransformer.src.tools.attention_visualization import visualize_layer, get_decoder_self
from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.core.batching import create_dataloaders, Batch
from algorithms.symbolicTransformer.src.tools.helper import load_config
from common.output_decoder import greedy_decode, beam_search
from common.metrics.bleu import compute_corpus_level_bleu_score


def check_outputs(
        model,
        vocab,
        dataloader_validation,
        n_examples=15,
        pad_idx=2,
        eos_string=Tag.STOP.value):

    results = [()] * n_examples
    for example_id in range(n_examples):

        print("\nExample %d ========\n" % example_id)

        # load a batch element
        data_val = next(iter(dataloader_validation))
        data_val_batch = Batch(data_val[0], data_val[1], pad_idx)

        # retrieve tokens with itos elements
        src_tokens = [
            vocab.vocab_src.get_itos()[x] for x in data_val_batch.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab.vocab_tgt.get_itos()[x] for x in data_val_batch.tgt[0] if x != pad_idx
        ]

        # pretty print source and target
        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )

        # choose decoder style from config
        if int(learning_configuration["beam_search"]) == 1:
            decoded = beam_search(model, data_val_batch.src,
                                     beam_size=int(learning_configuration["beam"]['beam-size']),
                                     max_decoding_time_step=int(learning_configuration["beam"]['max-decoding-time-step']))
            # top_hypothesis = [hyps[0] for hyps in hypothesis]
        else:
            decoded = greedy_decode(model, data_val_batch, 72, 0)[0]

        # pretty print the model output
        model_output = (
                " ".join(
                    [vocab.vocab_tgt.get_itos()[x] for x in decoded if x != pad_idx]
                ).split(eos_string, 1)[0]
                + eos_string
        )
        print("Model Output               : " + model_output.replace("\n", ""))

        # run BLEU score
        results[example_id] = (data_val_batch, src_tokens, tgt_tokens, decoded, model_output)
        hypothesis = []
        mo = model_output.split(" ")
        for h in mo:
            hypothesis.append(h)

        # bleu_score = compute_corpus_level_bleu_score(tgt_tokens, hypothesis)
        # print(f"BLEU score : {bleu_score*100} ---")

    return results


def run_model_example(config, n_examples=5):

    vocab = Vocab(load_tokenizers(), config, EnvType.DEV)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        vocab,
        torch.device("cpu"),
        application_path,
        batch_size=1,
        is_distributed=False
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab.vocab_src), len(vocab.vocab_tgt), config)
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

    model_learned, data_learned = run_model_example(config=learning_configuration)
    data_graph = data_learned[len(data_learned) - 1]

    chart = visualize_layer(
        model_learned, 1, get_decoder_self, len(data_graph[1]), data_graph[1], data_graph[1]
    )

    chart.save('output/translation_attention.html', embed_options={'renderer': 'svg'})
