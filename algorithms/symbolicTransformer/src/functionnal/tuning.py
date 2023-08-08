import numpy
import torch
import torch.nn as nn
import yaml
import datetime
import spacy

from common.metrics.processing_score import Translation

RUN_EXAMPLES = True

"""
Load config based on :
ProgressiveTransformersSLP by Ben Saunders
https://github.com/BenSaunders27/ProgressiveTransformersSLP
"""
def load_config(path="config.yaml") -> dict:
    """
    Loads and parses a YAML configuration file
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


"""
The tunings functionalities are coming from :
Annotated transformer
Huang, et al. 2022 / Rush, et al. 2019
nlp.seas.harvard.edu/annotated-transformer
"""

class LabelSmoothing(nn.Module):
    """
    Implement label smoothing with
    Kullback-Leibler divergence loss criterion
    :return: a criterion (detached tensor)
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def data_preparation(a_list, shuffling=False, join_data=False):
    """
    Divide the data into 1/3 validation and 2/3 training
    :param a_list: list of datas
    :param shuffling: shuffle data at the end
    :param join_data: data augmentation by joining
    :return: a training set and a validation set
    """

    tier = len(a_list)//3
    full_learning_set = []

    # add index to keep sentences context
    for i, k in enumerate(a_list):
        row = [i, k]
        full_learning_set.append(row)

    # split into training and validation
    if shuffling:
        numpy.random.shuffle(full_learning_set)

    validation_set = full_learning_set[:tier]  # one tiers
    training_set = full_learning_set[tier:]  # two tiers

    # sort training set
    training_set.sort()

    # data augmentation on training set (group phrases)
    if join_data:
        augmented_training_set = []
        for i in range(1, len(training_set)):
            new_row = [[], []]
            previous = training_set[i-1]
            current = training_set[i]
            if current[0] == (int(previous[0])+1):
                new_row[0] = str(previous[1][0])+" "+str(current[1][0])
                new_row[1] = str(previous[1][1])+" "+str(current[1][1])
                new_tab = [i, numpy.array([new_row[0], new_row[1]])]
                augmented_training_set.append(new_tab)

        augmented_training_set = augmented_training_set + training_set
    else:
        augmented_training_set = training_set

    # remove index
    prepared_training = []
    for k in augmented_training_set:
        data = [str(k[1][0]), str(k[1][1])]
        prepared_training.append(data)

    prepared_validation = []
    for k in validation_set:
        data = [str(k[1][0]), str(k[1][1])]
        prepared_validation.append(data)

    if shuffling:
        numpy.random.shuffle(prepared_training)

    return prepared_training, prepared_validation

def approximate_src(src):
    # nlp = spacy.load("fr_core_news_sm")
    print("approximate "+str(len(src))+" sources, start @ ", datetime.datetime.now())
    updated_src = []
    for i, s in enumerate(src):
        # us = [Translation.approximation(nlp, s[0]), s[1]]
        us = [Translation.approximation(s[0]), s[1]]
        updated_src.append(us)

    print("approximate "+str(len(src))+" sources, end @ ", datetime.datetime.now())
    return updated_src

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
