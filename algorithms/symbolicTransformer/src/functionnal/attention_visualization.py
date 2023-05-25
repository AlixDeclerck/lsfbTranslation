import torch
import matplotlib.pyplot as plt


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def plot_attention_maps(model, input_data, getter_fn):

    # selected attention
    attn_maps = getter_fn(model, 1)
    res = torch.squeeze(attn_maps)
    columns_nbr = res.size(dim=0)
    att_size = res.size(dim=1)

    fig, ax = plt.subplots(1, columns_nbr)

    for i, att_t in enumerate(torch.split(torch.squeeze(res, 0), 1, dim=0)):
        ax[i].imshow(torch.squeeze(att_t).detach().numpy(), origin='lower', vmin=0)
        ax[i].set_xticks(list(range(att_size)))
        ax[i].set_yticks(list(range(att_size)))
        ax[i].set_axis_off()
        # ax[i].invert_yaxis()
        # ax[i].set_xticklabels(input_data.tolist())
        # ax[i].set_yticklabels(input_data.tolist())
        # ax[i].set_title(f"Layer {i}, Head {i}")

    fig.subplots_adjust(hspace=0.5)
    plt.show()
