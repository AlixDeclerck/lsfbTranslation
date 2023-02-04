#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    training_the_system.py train --app-path=<file>
"""

import os
from os.path import exists

import GPUtil
import numpy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from common.constant import dir_separator
from docopt import docopt

from algorithms.data_loader.src.dal import EnvType

from algorithms.symbolicTransformer.src.core.batching import Batch, create_dataloaders
from algorithms.symbolicTransformer.src.tools.helper import DummyOptimizer, DummyScheduler
from algorithms.symbolicTransformer.src.core.loss_functions import SimpleLossCompute
from algorithms.symbolicTransformer.src.core.architecture import make_model, LabelSmoothing
from algorithms.symbolicTransformer.src.core.training import TrainState, rate, run_epoch
from algorithms.symbolicTransformer.src.core.data_preparation import load_tokenizers, Vocab
from algorithms.symbolicTransformer.src.tools.helper import load_config


def train_worker(
        gpu,
        ngpus_per_node,
        config,
        is_distributed=False):

    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    vocab_src = vocab.vocab_src
    vocab_tgt = vocab.vocab_tgt
    token_fr = vocab.french_tokens

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), config)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        token_fr,
        application_path,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["model_path"]+config["model_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        simple_loss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(simple_loss)
        torch.cuda.empty_cache()

    file_path = "%s%s%s" % (config["model_path"], config["model_prefix"], config["model_suffix"])
    torch.save(module.state_dict(), file_path)


def train_distributed_model(config):

    vocab_src = vocab.vocab_src
    vocab_tgt = vocab.vocab_tgt
    token_fr = vocab.french_tokens

    number_of_gpu = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {number_of_gpu}")
    print("Spawning training processes ...")
    numpy.spawn(
        train_worker,
        nprocs=number_of_gpu,
        args=(number_of_gpu, vocab_src, vocab_tgt, token_fr, config, True)
    )


def train_model(config):
    if config["distributed"]:
        train_distributed_model(
            config
        )
    else:
        train_worker(
            0, 1, config, False
        )


def load_trained_model(vocab_src_len, vocab_tgt_len, config):
    model_path = str(config["model_path"])+str(config["model_prefix"])+str(config["model_suffix"])
    if not exists(model_path):
        train_model(config)

    model = make_model(vocab_src_len, vocab_tgt_len, config)
    model.load_state_dict(torch.load(model_path))
    return model


# ---------------------------------------------------------------


if __name__ == '__main__':

    # CONFIGURATION
    args = docopt(__doc__)
    application_path = os.environ['HOME']+dir_separator+args['--app-path']+dir_separator
    learning_configuration = load_config()

    # PREPROCESSING
    vocab = Vocab(load_tokenizers(), learning_configuration, application_path, EnvType.DEV)

    # TRAINING
    trained_model = load_trained_model(len(vocab.vocab_src), len(vocab.vocab_tgt), learning_configuration)

    # OUTPUT
    print(trained_model)
