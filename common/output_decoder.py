import sys
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.constant import Hypothesis, start_symbol, Tag


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0
    # https://pytorch.org/docs/stable/generated/torch.triu.html
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def greedy_decode(model, data, max_len):

    # retrieve source sentence tokens from vocab
    src = data.src
    src_mask = data.src_mask

    # run the encoder
    encoder_output = model.encode(src, src_mask)

    # initialize
    estimation = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    hypotheses = [Hypothesis(value=[str(Tag.START.value)], score=0)]

    ok = True
    for i in range(max_len - 1):

        # decoder
        decoder_output = model.decode(
            encoder_output,                                             # src (memory)
            src_mask,                                                   # src_mask
            estimation,                                                 # tgt
            subsequent_mask(estimation.size(1)).type_as(src.data)       # tgt_mask
        )

        # output
        probabilities = model.generator(decoder_output[:, -1])

        # greedy selection of the most probable (next) word
        loss, next_token = torch.max(probabilities, dim=1)
        next_token = next_token.data[0]
        next_word = model.vocab.untokenize_tgt([next_token])
        if next_word[0] == str(Tag.STOP.value):
            ok = False

        if ok:
            hypotheses.append(Hypothesis(value=next_word, score=loss.data.numpy()[0]))

        # concatenate word to tensor
        estimation = torch.cat([estimation, torch.zeros(1, 1).type_as(src.data).fill_(next_token)], dim=1)

    hypotheses.append(Hypothesis(value=[str(Tag.STOP.value)], score=0))

    return hypotheses, estimation


"""
Original source :
CS224N 2019-20: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

def beam_search(model, data, config, beam_size, max_decoding_time_step):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model : NMT Model
    @param data : List of sentences (words) in source language, from test set.
    @param config : from configuration file
    @param beam_size : beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step : maximum sentence length that Beam search can produce
    @returns hypotheses : List of Hypothesis translations for every source sentence.
    """
    was_training = model.training   # boolean that says if the model was trained
    model.eval()                    # evaluation mode
    hypotheses = []                 # initialization

    # adapt batch list to one sentence case
    # sentences = tqdm(test_data_src, desc='Decoding', file=sys.stdout)  # multiple
    sentences = [data]
    # optimization (not include following in gradient)
    with torch.no_grad():

        # decode a test set
        for src_sent in sentences:

            # for each sentences
            example_hypothesis = model_beam_search(
                model,
                src_sent,
                config,
                beam_size=beam_size,
                max_decoding_time_step=max_decoding_time_step)

            # update hypothesis
            hypotheses.append(example_hypothesis)

    # update model status
    if was_training:
        model.train(was_training)

    # return Hypothesis object
    return hypotheses


def model_beam_search(model, data, config, beam_size: int = 5, max_decoding_time_step: int = 70):
    """ Given a single source sentence,
    perform beam search,
    yielding translations in the target language.

    @param model                    : learned model
    @param data                     : a single source sentence (words)
    @param config                   : learning configuration
    @param beam_size                : beam size
    @param max_decoding_time_step   : maximum number of time steps to unroll the decoding RNN

    @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
    """

    # initializations
    estimation = torch.zeros(1, 1).fill_(start_symbol).type_as(data.src.data)
    hypotheses = [[Tag.START.value]]
    hyp_scores = torch.ones(len(hypotheses), dtype=torch.float, device=model.device)
    completed_hypotheses = []
    last_decoder_layer = int(config["layers"]) - 1

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:

        # inverse counter counting and t
        live_hyp_num = beam_size - len(completed_hypotheses)
        t += 1

        # compute the decoder output based on the attention time_t
        decoder_output = model.decode(
            model.encode(data.src, data.src_mask),                           # src (memory)
            data.src_mask,                                                   # src_mask
            estimation,                                                      # tgt
            subsequent_mask(estimation.size(1)).type_as(data.src.data)       # tgt_mask
        )

        mha_att = torch.squeeze(model.decoder.layers[last_decoder_layer].self_attn.attn)

        # transform generator output to log probabilities of words
        log_p_t = F.log_softmax(model.generator(decoder_output[:, -1]))

        # construct tensor from log probabilities of words
        continuing_hyp_scores = torch.matmul(torch.unsqueeze(hyp_scores, 1), log_p_t).view(-1)

        # apply top k return score and word position
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(continuing_hyp_scores, k=live_hyp_num)

        # normalization
        prev_hyp_ids = top_cand_hyp_pos / len(model.vocab.tgt)
        hyp_word_ids = top_cand_hyp_pos % len(model.vocab.tgt)

        # initializations
        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()                    # previous hypotheses
            hyp_word_id = hyp_word_id.item()                    # hypothesis word id
            cand_new_hyp_score = cand_new_hyp_score.item()      # hypothesis candidate score

            # retrieve word from id
            hyp_word = model.vocab.tgt.get_itos()[hyp_word_id]

            # add new hypotheses to previous state in a new variable
            new_hyp_sent = hypotheses[int(prev_hyp_id)] + [hyp_word]

            # append or complete hypotheses
            if hyp_word == Tag.STOP.value:
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        # end loop condition
        if len(completed_hypotheses) == beam_size:
            break

        # tensor from the live "hypotheses id" (which gives the target potentiality size?)
        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=model.device)

        # update the estimation
        for h in live_hyp_ids:
            estimation = torch.cat([estimation, torch.zeros(1, 1).type_as(data.src.data).fill_(h)], dim=1)

        # update the attention
        mha_att_t = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(mha_att[live_hyp_ids], -2), -1), -1)
        model.decoder.layers[last_decoder_layer].self_attn.attn = mha_att_t

        # assign hypothesis with the new²
        hypotheses = new_hypotheses

        # create tensor from new hypothesis scores
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=model.device)

    # add hypothesis / score in Hypothesis format
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item()))

    # sort result
    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    # return Hypothesis object
    return completed_hypotheses


def beam_search_word2vec_id(model, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int): # -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model : NMT Model
    @param test_data_src : List of sentences (words) in source language, from test set.
    @param beam_size : beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step : maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training   # boolean that says if the model was trained
    model.eval()                    # evaluation mode
    hypotheses = []                 # initialization
    with torch.no_grad():           # optimization (not include following in gradient)

        # decode a test set
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):

            # for each sentences
            example_hypothesis = model_beam_search_word2vec_id(
                model,
                src_sent,
                beam_size=beam_size,
                max_decoding_time_step=max_decoding_time_step)

            # update hypothesis
            hypotheses.append(example_hypothesis)

    # update model status
    if was_training:
        model.train(was_training)

    # return Hypothesis object
    return hypotheses


def model_beam_search_word2vec_id(model, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70):  # -> List[Hypothesis]:
    """ Given a single source sentence, perform beam search, yielding translations in the target language.
    @param model : learned model
    @param src_sent : a single source sentence (words)
    @param beam_size : beam size
    @param max_decoding_time_step : maximum number of time steps to unroll the decoding RNN
    @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
    """
    # an input token words_size x 1 tensor
    src_sentences_var = model.vocab.src.to_input_tensor([src_sent], model.device)

    # encoder returns (1 x words_size x 2*embedding) size tensor and tuple of decoding vectors 1 x embedding
    src_encodings, dec_init_vec = model.encode(src_sentences_var, [len(src_sent)])

    # [attention adaptation : multi head / projection]
    # src_encodings_att_linear = self.att_projection(src_encodings)
    src_encodings_att_linear = model.att_multiHeaded(src_encodings)

    # tuple of decoding vectors 1 x embedding
    h_tm1 = dec_init_vec

    # empty 1 x 256 tensor
    att_tm1 = torch.zeros(1, model.hidden_size, device=model.device)

    # end phrase id
    eos_id = model.vocab.tgt['</s>']

    # initializations
    hypotheses = [['<s>']]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=model.device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:

        # counters
        t += 1
        hyp_num = len(hypotheses)

        # expand le src encoding à (hyp_num x word x 2*embedding)
        exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

        # expand attention to (hyp_num x word x embedding)
        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                       src_encodings_att_linear.size(1),
                                                                       src_encodings_att_linear.size(2))

        # retrieve a tensor made with target vocab
        y_tm1 = torch.tensor([model.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=model.device)

        # do an embedding from the tensor words (1x256)
        y_t_embed = model.model_embeddings.target(y_tm1)

        # concat on dim 2 the decoder inputs
        x = torch.cat([y_t_embed, att_tm1], dim=-1)

        # Compute one forward step of the LSTM decoder, including the attention computation.
        (h_t, cell_t), att_t, _ = model.step(x,
                                             h_tm1,
                                             exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             enc_masks=None)

        # log probabilities over target words
        log_p_t = F.log_softmax(model.target_vocab_projection(att_t), dim=-1)

        # inverse counter
        live_hyp_num = beam_size - len(completed_hypotheses)

        # construct tensor from log probabilities of words
        continuing_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)

        # apply top k return score and word position
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(continuing_hyp_scores, k=live_hyp_num)

        # normalization
        prev_hyp_ids = top_cand_hyp_pos / len(model.vocab.tgt)
        hyp_word_ids = top_cand_hyp_pos % len(model.vocab.tgt)

        # initializations
        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()                    # previous hypotheses
            hyp_word_id = hyp_word_id.item()                    # hypothesis word id
            cand_new_hyp_score = cand_new_hyp_score.item()      # hypothesis candidate score

            # retrieve word from id
            hyp_word = model.vocab.tgt.id2word[hyp_word_id]

            # add new hypotheses to previous state in a new variable
            new_hyp_sent = hypotheses[int(prev_hyp_id)] + [hyp_word]

            # append or complete hypotheses
            if hyp_word == '</s>':
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1], score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        # end loop condition
        if len(completed_hypotheses) == beam_size:
            break

        # tensor from the live "hypotheses id"
        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=model.device)

        # assign the tuple hidden and cell from live_hyp_ids
        h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])

        # update model's attention from "live hypothesis id"
        att_tm1 = att_t[live_hyp_ids]

        # assign hypothesis with the new
        hypotheses = new_hypotheses

        # create tensor from new hypothesis scores
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=model.device)

    # add hypothesis / score in Hypothesis format
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:], score=hyp_scores[0].item()))

    # sort result
    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    # return Hypothesis object
    return completed_hypotheses
