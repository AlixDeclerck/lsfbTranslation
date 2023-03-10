#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
import copy
import math
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """
    Produce N identical layers.
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, hidden_size, h=1, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # self.linears = copy.deepcopy(nn.Linear(d_model, d_model)) # is not iterable?
        self.linears = clones(nn.Linear(d_model, hidden_size), 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        query = x
        key = x
        value = x

        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = [
        #     lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #     for lin, x in zip(self.linears, (query, key, value))
        # ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # Coding
        self.encoder = nn.LSTM(embed_size, self.hidden_size, bias=True, bidirectional=True, batch_first=True)  # (Bidirectional LSTM with bias)
        self.decoder = nn.LSTMCell(embed_size + hidden_size, self.hidden_size, bias=True)  # (LSTM Cell with bias)
        self.h_projection = nn.Linear(embed_size * 2, hidden_size, bias=False)  # (Linear Layer with no bias), called W_{h} in the PDF
        self.c_projection = nn.Linear(embed_size * 2, hidden_size, bias=False)  # (Linear Layer with no bias), called W_{c} in the PDF

        # Basic dot product attention
        self.att_projection = nn.Linear(embed_size * 2, hidden_size, bias=False)  # (Linear Layer with no bias), called W_{attProj} in the PDF
        self.att_multiHeaded = MultiHeadedAttention(embed_size * 2, hidden_size)

        self.combined_output_projection = nn.Linear(embed_size * 3, hidden_size, bias=False)  # (Linear Layer with no bias), called W_{u} in the PDF
        self.target_vocab_projection = nn.Linear(hidden_size, len(self.vocab.tgt), bias=False)  # (Linear Layer with no bias), called W_{vocab} in the PDF
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)  # (Dropout Layer)

        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0

    def forward(self, source: List[List[str]], target: List[List[str]]):  # -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)  # Apply the encoder to `source_padded` by calling `self.encode()`
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)  # Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)  # Apply the decoder to compute combined-output by calling `self.decode()`
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)  # Compute log probability distribution over the target vocabulary using the combined_outputs returned by the `self.decode()` function.

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]):  # -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """

        X = self.model_embeddings.source(source_padded)  # Tensor `X` with shape (src_len, b, e) - src_len = maximum source sentence length, b = batch size, e = embedding size. Note that there is no initial hidden state or cell for the encoder.
        packed_seq = pack_padded_sequence(X, source_lengths)  # apply the `pack_padded_sequence` function to X
        enc_hiddens, (last_hidden, last_cell) = self.encoder(packed_seq)  # apply encoder
        enc_hiddens = pad_packed_sequence(sequence=enc_hiddens)[0].permute(1, 0, 2)  # apply the `pad_packed_sequence` function to enc_hiddens

        # `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards. Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        # Apply the h_projection layer to this in order to compute init_decoder_hidden. This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), 1))  # vector concatenation

        # `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards. Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        # "Apply the c_projection layer to this in order to compute init_decoder_cell. This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), 1))

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor, dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor):  # -> torch.Tensor:

        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """

        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        # [adapt attention]
        # enc_hiddens_proj = self.att_projection(enc_hiddens)  # Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`, which should be shape (b, src_len, h), where b = batch size, src_len = maximum source length, h = hidden size. This is applying W_{attProj} to h^enc, as described in the PDF.
        enc_hiddens_proj = self.att_multiHeaded(enc_hiddens)

        Y = self.model_embeddings.target(target_padded)  # Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings. where tgt_len = maximum target sentence length, b = batch size, e = embedding size.

        for Y_t in torch.split(Y, 1, dim=0):  # Use the torch.split function to iterate over the time dimension of Y. Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
            Y_t = torch.squeeze(Y_t)  # Squeeze Y_t into a tensor of dimension (b, e).
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)  # Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
            _, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)  # Use the step function to compute the the Decoder's next (cell, state) values as well as the new combined output o_t.
            combined_outputs.append(o_t)  # Append o_t to combined_outputs
            o_prev = o_t  # Update o_prev to the new o_t.

        combined_outputs = torch.stack(combined_outputs, dim=0)  # Use torch.stack to convert combined_outputs from a list length tgt_len of tensors shape (b, h), to a single tensor shape (tgt_len, b, h) where tgt_len = maximum target sentence length, b = batch size, h = hidden size.

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor, dec_state: Tuple[torch.Tensor, torch.Tensor], enc_hiddens: torch.Tensor, enc_hiddens_proj: torch.Tensor, enc_masks: torch.Tensor):  # -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """
        dec_state = self.decoder(Ybar_t, dec_state)  # Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        dec_hidden, dec_cell = dec_state  # Split dec_state into its two parts (dec_hidden, dec_cell)
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2)), 2)  # Compute the attention scores e_t, a Tensor shape (b, src_len). Note: b = batch_size, src_len = maximum source length, h = hidden size.

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=1)  # Apply softmax to e_t to yield alpha_t
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens), 1)

        U_t = torch.cat((dec_hidden, a_t), 1)  # Concatenate dec_hidden with a_t to compute tensor U_t
        V_t = self.combined_output_projection(U_t)  # Apply the combined output projection layer to U_t to compute tensor V_t
        O_t = self.dropout(torch.tanh(V_t))  # Compute tensor O_t by first applying the Tanh function and then the dropout layer.

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]):  # -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size, src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
