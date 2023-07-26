#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import os
from pathlib import Path

import altair as alt
import pandas
import torch
from docopt import docopt

from algorithms.symbolicTransformer.src.core.architecture import NMT
from algorithms.symbolicTransformer.src.core.batching import Batch, collate_batch
from algorithms.symbolicTransformer.src.functionnal.data_preparation import retrieve_conte_dataset, Vocab
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import EnvType, Dialect, Corpus, HypothesisType, d_date
from common.metrics.bleu_score import Translation
from common.output_decoder import greedy_decode, beam_search


def run_inference(config, vocab, test_dataset):
    today = d_date()
    path = "../../../common/output/decoding_scores_"+today+".csv"
    filepath = Path(path)
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

    model = NMT(vocab, config)
    model.load_state_dict(
        torch.load(config["configuration_path"]["model_path"]+config["configuration_path"]["model_prefix"]+config["configuration_path"]["model_suffix"], map_location=torch.device("cpu"))
    )

    # Filter datas from english or glosses depends on the config
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
        if is_beam_search:
            trans.add_hypothesis(HypothesisType.BEAM, hypothesis_beam)
        trans.add_hypothesis(HypothesisType.GREEDY, hypothesis_greedy)
        trans.display_translation(title)
        trans.export(title, df_scores)

        # CONSTRUCT RESULT VALUE LIST
        results.append([
            data_val_batch,
            trans,
            estimation_greedy])

        if i+1 == limit:
            df_scores.to_csv(filepath)
            return model, results

    df_scores.to_csv(filepath)
    return model, results


# -------------------------------------------------------------------------------------


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """convert a dense matrix to a data frame with row and column indices"""
    return pandas.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )

def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


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

    # Load Vocabulary
    saved_vocab = Vocab(learning_configuration)
    saved_vocab.retrieve_from_disk()
    print("Vocab loaded.. ")
    print("Source contains "+str(len(saved_vocab.src))+" elements")
    print("Target contains "+str(len(saved_vocab.tgt))+" elements")
    print()

    # Load test set
    loaded_test_set = retrieve_conte_dataset(
        EnvType.TEST.value,
        application_path,
        learning_configuration["configuration_path"]["selected_db"],
        Dialect.LSF,
        saved_vocab.is_english_output, False, 10000)
    print("Test set loaded.. ")
    print("With "+str(len(loaded_test_set))+" elements")
    print()

    # INFERENCE
    used_model, inferred_data = run_inference(config=learning_configuration, vocab=saved_vocab, test_dataset=loaded_test_set)

    row_example = inferred_data[0][1].source_text.split(" ")
    col_example = inferred_data[0][1].greedy_hypothesis.split(" ")
    # col_example = inferred_data[0][1].source_text.split(" ")
    length = max(len(row_example), len(col_example))

    layer_viz = [
        visualize_layer(
            used_model, layer, get_decoder_self, length, row_example, col_example
        )
        for layer in range(6)
    ]
    chart = alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )

    # chart.save('output/learning_ST_decoder_cross_2023-07-26.html', embed_options={'renderer': 'svg'})
