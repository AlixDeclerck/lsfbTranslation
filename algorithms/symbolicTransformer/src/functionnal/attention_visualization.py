import torch
import matplotlib.pyplot as plt
import math

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


def plot_attention_maps(model, input_data, nbr_attentions):
    """
    todo: input_data have to a nice class with reference and all hypothesis + detail + infos
    We retrieve the attentions in the model using the 3 functions above
    :param model: model that was used during inference
    :param input_data: an object with all inferred values
    :param nbr_attentions: nombre of attentions used by the architecture of the model
    """
    limit = nbr_attentions-1
    plot_attention_map(model, input_data, get_decoder_self, [0, math.floor(limit/2), limit])

def plot_attention_map(model, input_data, getter_fn, att_to_display):
    """
    We retrieve a specific attention matrix
    """
    # selected attention
    attn = torch.squeeze(getter_fn(model, 1))
    att_size = attn.size(dim=1)
    selected_att = []

    for i in att_to_display:
        selected_att.append(torch.squeeze(torch.split(attn, 1, dim=0)[i]))

    fig, ax = plt.subplots(1, len(att_to_display))

    for i, att_t in enumerate(selected_att):
        ax[i].imshow(torch.squeeze(att_t).detach().numpy(), origin='lower', vmin=0)
        ax[i].set_xticks(list(range(att_size)))
        ax[i].set_yticks(list(range(att_size)))
        ax[i].set_axis_off()

    fig.subplots_adjust(hspace=0.5)
    plt.show()

