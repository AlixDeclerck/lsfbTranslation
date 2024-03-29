import math

import matplotlib.pyplot as plt
import torch

from common.constant import Case, d_date

"""
The attention visualization was initially inspired from :
Annotated transformer
Huang, et al. 2022 / Rush, et al. 2019
nlp.seas.harvard.edu/annotated-transformer

We recreated attentions output using matplotlib instead of altair
"""


def get_encoder(model, layer):
    """
    get the encoder's attention
    :param model: model that was used during inference
    :param layer: a selected layer
    :return: the attention at a specified layer from the model
    """
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    """
    get the decoder's self attention
    :param model: model that was used during inference
    :param layer: a selected layer
    :return: the attention at a specified layer from the model
    """
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    """
    get the decoder's mixed with the source's attention
    :param model: model that was used during inference
    :param layer: a selected layer
    :return: the attention at a specified layer from the model
    """
    return model.decoder.layers[layer].src_attn.attn


def plot_attention_maps(model, input_data, cfg):
    """
    todo: input_data have to a nice class with reference and all hypothesis + detail + infos
    We retrieve the attentions in the model using the 3 functions above
    :param model: model that was used during inference
    :param input_data: an object with all inferred values
    :param cfg: config file
    """
    nbr_attentions = cfg["hyper_parameters"]["h_attention_layers"]
    limit = nbr_attentions-1
    txt_translation = input_data[0][1]

    plot_attention_map(model, txt_translation, get_decoder_self, [0, math.floor(limit/2), limit])


def plot_attention_map(model, txt_translation, getter_fn, att_to_display):
    """
    We retrieve a specific attention matrix
    """
    resizing_img_coef = 10
    source_size = len(txt_translation.source_text.split(" "))
    target = txt_translation.greedy_hypothesis.split(" ")
    target = [i for i in target if i not in ["", " "]]
    target_size = len(target)

    case = Case.FIRST
    today = d_date()
    add = "SF_"
    filename = "../../../common/img/ATT_ST_"+today+"_"+str(add)+str(case.value[1])+".png"
    attn = torch.squeeze(getter_fn(model, 8))
    att_size = attn.size(dim=1)
    selected_att = []

    for i in att_to_display:
        selected_att.append(torch.squeeze(torch.split(attn, 1, dim=0)[i]))

    fig, ax = plt.subplots(1, len(att_to_display))

    for i, att_t in enumerate(selected_att):
        attention_matrix = torch.squeeze(att_t).detach().numpy()
        # attention_matrix = attention_matrix[0:source_size, 0:target_size]
        # print(numpy.shape(attention_matrix))
        ax[i].imshow(attention_matrix, vmin=0, origin='lower')  # , vmax=5
        ax[i].set_xticks(list(range(att_size)))
        ax[i].set_yticks(list(range(att_size)))
        ax[i].set_axis_off()
        ax[i].invert_yaxis()
        # ax[i].title("txt")
        # ax[i].figure(figsize=(100, 100))

    fig.subplots_adjust(hspace=0.5)
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # figsize=(source_size*resizing_img_coef, target_size*resizing_img_coef)
    plt.figure()
    plt.savefig(filename)
    plt.show()

    # https://matplotlib.org/stable/gallery/statistics/hist.html
    # figsize : fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True, sharey=True, tight_layout=True)
