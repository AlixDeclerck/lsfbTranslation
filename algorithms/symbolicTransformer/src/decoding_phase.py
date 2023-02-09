#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py train --app-path=<file>
"""

import torch
from common.constant import dir_separator
from docopt import docopt
import os

from algorithms.data_loader.src.dal import EnvType
from algorithms.symbolicTransformer.src.core.architecture import make_model
from algorithms.symbolicTransformer.src.tools.attention_visualization import visualize_layer, get_decoder_self
from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.core.batching import create_dataloaders
from algorithms.symbolicTransformer.src.core.batching import Batch
from common.output_decoder import greedy_decode, beam_search
from algorithms.symbolicTransformer.src.tools.helper import load_config
from common.metrics.bleu import compute_corpus_level_bleu_score


def check_outputs(
        valid_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        n_examples=15,
        pad_idx=2,
        eos_string="</s>"):

    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )

        if int(learning_configuration["beam_search"]) == 1:
            decoded = beam_search(model, rb.src,
                                     beam_size=int(learning_configuration["beam"]['beam-size']),
                                     max_decoding_time_step=int(learning_configuration["beam"]['max-decoding-time-step']))
        else:
            decoded = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]

        model_output = (
                " ".join(
                    [vocab_tgt.get_itos()[x] for x in decoded if x != pad_idx]
                ).split(eos_string, 1)[0]
                + eos_string
        )
        print("Model Output               : " + model_output.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, decoded, model_output)

        hypothesis = []
        mo = model_output.split(" ")
        for h in mo:
            hypothesis.append(h)
        top_hypothesis = [hyps[0] for hyps in hypothesis]
        bleu_score = compute_corpus_level_bleu_score(tgt_tokens, hypothesis)
        print(f"BLEU score : {bleu_score*100} ---")

    return results


def run_model_example(config, n_examples=5):

    token_fr = load_tokenizers()
    vocab = Vocab(token_fr, config, EnvType.DEV)

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab.vocab_src,
        vocab.vocab_tgt,
        token_fr,
        application_path,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab.vocab_src), len(vocab.vocab_tgt), config)
    model.load_state_dict(
        torch.load(config["model_path"]+config["model_prefix"]+config["model_suffix"], map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab.vocab_src, vocab.vocab_tgt, n_examples=n_examples
    )
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
