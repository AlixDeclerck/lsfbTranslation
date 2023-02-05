import time
import GPUtil
import numpy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from os.path import exists
import os
from common.constant import Tag

from algorithms.symbolicTransformer.src.core.batching import Batch, create_dataloaders
from algorithms.symbolicTransformer.src.tools.helper import DummyOptimizer, DummyScheduler
from algorithms.symbolicTransformer.src.core.loss_functions import SimpleLossCompute
from algorithms.symbolicTransformer.src.core.architecture import make_model, LabelSmoothing


def load_or_train_model(vocab, config):
    model_path = str(config["model_path"])+str(config["model_prefix"])+str(config["model_suffix"])
    if not exists(model_path):
        train_model(vocab, config)

    model = make_model(len(vocab.vocab_src), len(vocab.vocab_tgt), config)
    model.load_state_dict(torch.load(model_path))
    return model


def train_model(vocab, config):
    if config["distributed"]:
        train_distributed_model(vocab, config)
    else:
        train_worker(
            0, 1, vocab, config, False
        )


def train_distributed_model(vocab, config):

    number_of_gpu = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {number_of_gpu}")
    print("Spawning training processes ...")
    numpy.spawn(
        train_worker,
        nprocs=number_of_gpu,
        args=(number_of_gpu, vocab.vocab_src, vocab.vocab_tgt, vocab.french_tokens, config, True)
    )


class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0           # Steps in the current epoch
    accum_step: int = 0     # Number of gradient accumulation steps
    samples: int = 0        # total # of examples used
    tokens: int = 0         # total # of tokens processed


def train_worker(
        gpu,
        ngpus_per_node,
        vocab,
        config,
        model_saving_strategy=False,
        is_distributed=False):

    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab.vocab_tgt[Tag.BLANK.value]
    d_model = 512
    model = make_model(len(vocab.vocab_src), len(vocab.vocab_tgt), config)
    model.cuda(gpu)
    module = model
    is_main_process = True

    # distributed case
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # label smoothing
    criterion = LabelSmoothing(
        size=len(vocab.vocab_tgt), padding_idx=pad_idx, smoothing=config["target_label_smoothing"]
    )
    criterion.cuda(gpu)

    # dataloaders
    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab.vocab_src,
        vocab.vocab_tgt,
        vocab.french_tokens,
        application_path=config["application_path"],
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["base_lr"],                               # learning rate (default: 1e-3)
        betas=(0.9, 0.98),                                  # coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps=1e-9,                                           # term added to the denominator to improve numerical stability
        weight_decay=config["optimizer_weight_decay"],      # weight decay (L2 penalty) (default: 0)
        amsgrad=False                                       # AMSGrad variant (default: False)
    )

    # scheduler
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )

    # train state
    train_state = TrainState()

    # training by epochs
    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        # training
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            accum_step=config["accum_step"],
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        # saving module
        GPUtil.showUtilization()
        if is_main_process and model_saving_strategy:
            file_path = "%s%.2d.pt" % (config["model_path"]+config["model_prefix"], epoch)
            torch.save(module.state_dict(), file_path)

        torch.cuda.empty_cache()

        # model evaluation
        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        simple_loss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            accum_step=config["accum_step"],
            mode="eval",
        )

        # loss function in evaluation mode
        print(simple_loss)
        torch.cuda.empty_cache()

    # todo: save trained result and be able to improve
    if model_saving_strategy:
        file_path = "%s%s%s" % (config["model_path"], config["model_prefix"], config["model_suffix"])
        torch.save(module.state_dict(), file_path)


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        accum_step,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.n_tokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.n_tokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens
        if i % accum_step == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.n_tokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
