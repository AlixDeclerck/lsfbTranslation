import torch

from torch.nn.functional import pad
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from algorithms.symbolicTransformer.src.tools.helper import tokenize, split_list
from algorithms.symbolicTransformer.src.core.data_preparation import retrieve_phoenix_dataset
from algorithms.symbolicTransformer.src.core.data_preparation import Vocab
from common.output_decoder import subsequent_mask
from algorithms.data_loader.src.dal import EnvType
from common.constant import pad_idx


def collate_batch(
        batch,
        vocab: Vocab,
        device,
        max_padding=128,
        pad_id=2):

    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocab.src(vocab.tokenize_fr(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    vocab.tgt(vocab.tokenize_fr(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    return src, tgt


def create_dataloaders(
        vocab,
        device,
        application_path,
        batch_size=12000,
        max_padding=128,
        is_distributed=True):

    # def tokenize_fr(text):
    #     return tokenize(text, vocab.french_tokens)

    def collate_fn(batch):
        return collate_batch(
            batch,
            vocab,
            device,
            max_padding=max_padding,
            pad_id=vocab.src.get_stoi()["<blank>"],
        )

    # Dataset that will do the batches
    full = retrieve_phoenix_dataset(EnvType.TEST, application_path)
    train_iter, tmp_iter = split_list(full)

    test_iter, valid_iter = split_list(tmp_iter)

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None):
        self.src = src
        self.src_mask = (src != int(pad_idx)).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, int(pad_idx))
            self.n_tokens = (self.tgt_y != int(pad_idx)).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
