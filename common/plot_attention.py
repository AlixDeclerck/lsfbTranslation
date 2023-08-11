#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    decoding_phase.py cpu --app-path=<file>
"""

import os

import altair as alt
import pandas
import torch
from docopt import docopt

from algorithms.symbolicTransformer.src.decoding_phase import run_inference
from algorithms.symbolicTransformer.src.functionnal.tuning import load_config
from common.constant import d_date, current_session


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
    case = current_session()[0]
    args = docopt(__doc__)
    learning_configuration = load_config()
    application_path = os.environ['HOME'] + learning_configuration["configuration_path"]["application_path"] + args['--app-path'] + learning_configuration["configuration_path"]["application_path"]
    torch.cuda.empty_cache()
    today = d_date()

    if not args['cpu']:
        learning_configuration["learning_config"]["using_gpu"] = True
    else:
        learning_configuration["learning_config"]["using_gpu"] = False

    # INFERENCE
    used_model, inferred_data = run_inference(config=learning_configuration, app=application_path, save_file=False)

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
    file_saving = "../../../common/output/attention_"+today+"_"+str(case.value[1])+".html"
    chart.save(file_saving, embed_options={'renderer': 'svg'})
