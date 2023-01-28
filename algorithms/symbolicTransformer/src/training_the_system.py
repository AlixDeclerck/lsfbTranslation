import os
from os.path import exists

import GPUtil
import numpy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from algorithms.symbolicTransformer.src.core.batching import Batch, create_dataloaders
from algorithms.symbolicTransformer.src.tools.helper import DummyOptimizer, DummyScheduler
from algorithms.symbolicTransformer.src.core.loss_functions import SimpleLossCompute
from algorithms.symbolicTransformer.src.core.architecture import make_model, LabelSmoothing
from algorithms.symbolicTransformer.src.core.training import TrainState, rate, run_epoch
from algorithms.symbolicTransformer.src.core.vocabulary_builder import load_tokenizers, load_vocab
from algorithms.symbolicTransformer.src.tools.helper import load_config


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        config,
        is_distributed=False):

    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

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
        spacy_de,
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


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, config):

    number_of_gpu = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {number_of_gpu}")
    print("Spawning training processes ...")
    numpy.spawn(
        train_worker,
        nprocs=number_of_gpu,
        args=(number_of_gpu, vocab_src, vocab_tgt, spacy_de, config, True)
    )


def train_model(vocab_src, vocab_tgt, spacy_de, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, config, False
        )


def load_trained_model(vocab_src, vocab_tgt, spacy_de, config):
    model_path = str(config["model_path"])+str(config["model_prefix"])+str(config["model_suffix"])
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, config)

    model = make_model(len(vocab_src), len(vocab_tgt), config)
    model.load_state_dict(torch.load(model_path))
    return model


# ---------------------------------------------------------------

learning_configuration = load_config()

token_de = load_tokenizers()
src_vocabulary, tgt_vocabulary = load_vocab(token_de, learning_configuration)

trained_model = load_trained_model(src_vocabulary, tgt_vocabulary, token_de, learning_configuration)
print(trained_model)
