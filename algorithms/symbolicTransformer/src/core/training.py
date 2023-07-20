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
from common.persist_result import persist_training, persist_validation

from algorithms.symbolicTransformer.src.core.batching import Batch, create_dataloaders
from algorithms.symbolicTransformer.src.functionnal.tuning import DummyOptimizer, DummyScheduler, LabelSmoothing
from algorithms.symbolicTransformer.src.functionnal.loss_functions import SimpleLossCompute
from algorithms.symbolicTransformer.src.core.architecture import NMT

"""
The training file contents are coming from :
Annotated transformer
Huang, et al. 2022 / Rush, et al. 2019
nlp.seas.harvard.edu/annotated-transformer
"""

def load_or_train_model(vocab, environment, config):
    model_path = str(config["configuration_path"]["model_path"])+str(config["configuration_path"]["model_prefix"])+str(config["configuration_path"]["model_suffix"])
    if not exists(model_path):
        train_model(vocab, environment, config)

    model = NMT(vocab, config)
    model.load_state_dict(torch.load(model_path))
    return model


def train_model(vocab, environment, config):
    if config["learning_config"]["distributed"]:
        train_distributed_model(vocab, config)
    else:
        train_worker(
            gpu=0,
            ngpus_per_node=1,
            vocab=vocab,
            environment=environment,
            config=config,
            model_saving_strategy=True,
            is_distributed=False
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
        args=(number_of_gpu, vocab, config, True, True)
    )


class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0           # Steps in the current epoch
    accum_step: int = 0     # Number of gradient accumulation steps
    samples: int = 0        # total # of examples used
    tokens: int = 0         # total # of tokens processed


def train_worker(gpu, ngpus_per_node, vocab, environment, config, model_saving_strategy=False, is_distributed=False):

    persist_learning_measure = config["learning_config"]["persist_learning_measure"]

    if not persist_learning_measure:
        print(f"Train worker process using GPU: {gpu} for training", flush=True)

    torch.cuda.set_device(gpu)
    pad_idx = vocab.tgt[Tag.BLANK.value[0]]
    d_model = config["hyper_parameters"]["dimension"]
    model = NMT(vocab, config)
    model.cuda(gpu)
    module = model
    is_main_process = True

    # MULTI GPU CASE
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    # LABEL SMOOTHING
    criterion = LabelSmoothing(
        size=len(vocab.tgt), padding_idx=pad_idx, smoothing=config["hyper_parameters"]["target_label_smoothing"]
    )
    criterion.cuda(gpu)

    # BATCH SAMPLING
    train_dataloader, valid_dataloader = create_dataloaders(
        vocab,
        environment,
        gpu,
        english_output=config["learning_config"]["english_output"],
        application_path=config["configuration_path"]["application_path"],
        selected_db=config["configuration_path"]["selected_db"],
        batch_size=config["hyper_parameters"]["batch_size"] // ngpus_per_node,
        max_padding=config["learning_config"]["max_padding"],
        is_distributed=is_distributed,
        shuffling=config["learning_config"]["shuffling"]
    )

    # OPTIMIZATION
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_optimization"]["base_lr"]),
        betas=(float(config["learning_optimization"]["adam_optimizer_betas_1"]), float(config["learning_optimization"]["adam_optimizer_betas_2"])),
        eps=float(config["learning_optimization"]["adam_optimizer_eps"]),
        weight_decay=float(config["hyper_parameters"]["weight_decay"]),
        amsgrad=bool(config["learning_optimization"]["adam_optimizer_amsgrad"])
    )

    # SCHEDULER
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["learning_optimization"]["warmup"]
        ),
    )

    # TRAINING
    train_state = TrainState()
    smallest_loss = float('inf')
    for epoch in range(config["hyper_parameters"]["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        # training
        model.train()
        if not persist_learning_measure:
            print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)

        _, train_state = run_epoch(
            (Batch(b[0], b[1]) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion, config),
            optimizer,
            lr_scheduler,
            accum_step=config["hyper_parameters"]["accum_step"],
            mode="train+log",
            accum_iter=config["hyper_parameters"]["accum_iter"],
            train_state=train_state,
        )

        torch.cuda.empty_cache()

        # MODEL EVALUATION
        if not persist_learning_measure:
            print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)

        model.eval()
        simple_loss = run_epoch(
            (Batch(b[0], b[1]) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion, config),
            DummyOptimizer(),
            DummyScheduler(),
            accum_step=config["hyper_parameters"]["accum_step"],
            mode="eval",
        )

        # saving module
        GPUtil.showUtilization()
        if is_main_process and model_saving_strategy:
            actual_loss = simple_loss[0].item()
            if actual_loss < smallest_loss:
                smallest_loss = actual_loss
                file_path = "%s%.2d.pt" % (config["configuration_path"]["model_path"]+config["configuration_path"]["model_prefix"], epoch)
                torch.save(module.state_dict(), file_path)

        # loss function in evaluation mode
        if persist_learning_measure:
            persist_validation(epoch, simple_loss[0].item(), smallest_loss)
        else:
            print(str(simple_loss) + " , smallest_loss: " + str(smallest_loss))

        # gpu flush
        torch.cuda.empty_cache()

    if model_saving_strategy:
        file_path = "%s%s%s" % (config["configuration_path"]["model_path"], config["configuration_path"]["model_prefix"], config["configuration_path"]["model_suffix"])
        torch.save(module.state_dict(), file_path)


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, accum_step, mode="train", accum_iter=1, train_state=TrainState(), persist_learning_measure=False):

    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):

        training_result = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )

        # loss_node is used during backprop (training)
        loss, loss_node = loss_compute(training_result, batch.tgt_y, batch.n_tokens)

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

        # update loss & tokens value
        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens

        # Display i, n_accum, loss / batch.n_tokens, tokens / elapsed, lr
        if i % accum_step == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            if persist_learning_measure:
                normalized_loss = loss / batch.n_tokens
                tokens_time_processing = tokens / elapsed
                persist_training(epoch=i, acc=n_accum, loss=normalized_loss, tokens=tokens_time_processing, lr=lr)

            else:
                print("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f | Tokens / Sec: %7.1f | Learning Rate: %6.1e" % (i, n_accum, loss / batch.n_tokens, tokens / elapsed, lr))

            start = time.time()
            tokens = 0

        # delete this (batch iteration) loss objects
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
