import numpy
import torch
import matplotlib.pyplot as plt
import math
from common.constant import Case, d_date, Translation
# from common.metrics.bleu_score import Translation

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

    source_text = input_data[0][1]
    beam_hypothesis = input_data[0][4]
    greedy_hypothesis = input_data[0][5]
    reference = input_data[0][2]

    txt_translation = Translation(source_text=source_text, beam_hypothesis=beam_hypothesis, greedy_hypothesis=greedy_hypothesis, reference=reference)

    plot_attention_map(model, txt_translation, get_decoder_self, [0, math.floor(limit/2), limit])

def plot_attention_map(model, txt_translation, getter_fn, att_to_display):
    """
    We retrieve a specific attention matrix
    """
    resizing_img_coef = 10
    source_size = len(txt_translation.source_text)
    target_size = len(txt_translation.beam_hypothesis)

    case = Case.FIRST
    today = d_date()
    add = "SF_"
    filename = "../../../common/img/ATT_ST_"+today+"_"+str(add)+str(case.value[1])+".png"
    attn = torch.squeeze(getter_fn(model, 1))
    att_size = attn.size(dim=1)
    selected_att = []

    for i in att_to_display:
        selected_att.append(torch.squeeze(torch.split(attn, 1, dim=0)[i]))

    fig, ax = plt.subplots(1, len(att_to_display))

    for i, att_t in enumerate(selected_att):
        attention_matrix = torch.squeeze(att_t).detach().numpy()
        attention_matrix = attention_matrix[0:source_size, 0:target_size]
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
    plt.figure(figsize=(source_size*resizing_img_coef, target_size*resizing_img_coef))
    plt.savefig(filename)
    plt.show()
