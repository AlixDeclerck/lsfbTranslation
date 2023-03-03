import pandas
import altair
import scikitplot as skplt
import torch
# from sklearn.metrics import rfr
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy
import math

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """
    convert a dense matrix to a data frame with row and column indices
    """

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
        altair.Chart(data=df)
        .mark_rect()
        .encode(
            x=altair.X("col_token", axis=altair.Axis(title="")),
            y=altair.Y("row_token", axis=altair.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


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
    return altair.vconcat(
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


def plot_attention_maps(model, input_data, getter_fn, idx=0):

    col = 4

    # selected attention
    attn_maps = getter_fn(model, 1)

    if input_data is None:
        attentions = numpy.arange(attn_maps[0][idx].shape[-1])
    else:
        attentions = torch.split(torch.squeeze(attn_maps, 0), 1, dim=1)
        #  attentions = torch.squeeze(torch.split(torch.squeeze(attn_maps, 0), 1, dim=1)[1], 1)

    att_size = len(attentions)
    ln = int(math.ceil(att_size / col))

    fig, ax = plt.subplots(ln, col)

    for row in range(ln):
        for column in range(col):
            if (row+column) < att_size:
                att = torch.squeeze(attentions[row+column], 1)
                ax[row][column].imshow(att, origin='lower', vmin=0)
                ax[row][column].set_xticks(list(range(att_size)))
                # ax[row][column].set_xticklabels(input_data.tolist())
                ax[row][column].set_yticks(list(range(att_size)))
                # ax[row][column].set_yticklabels(input_data.tolist())
                ax[row][column].set_title(f"Layer {row+1}, Head {column+1}")

    fig.subplots_adjust(hspace=0.5)
    plt.show()

    # for attention in attentions:
    #
    #
    # plot_size = (19, 9)
    # plt.figure(figsize=plot_size)
    #
    # plt.title("title")
    # plt.gca().set_xlabel("x")
    # plt.gca().set_ylabel("y")
    # plt.imshow(attentions, origin='lower', vmin=0)
    # plt.gca().legend()
    # plt.show()
