import torch
import yaml

RUN_EXAMPLES = True


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]


def show_example(fn, args=[]):
    if RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


# source : https://github.com/BenSaunders27/ProgressiveTransformersSLP
def load_config(path="config.yaml") -> dict:
    """
    Loads and parses a YAML configuration file
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


# source : https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/misc.py
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm)
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.contiguous().view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
