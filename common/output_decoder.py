import sys
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.constant import Hypothesis, Tag


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
    estimation = torch.zeros(1, 1).fill_(Tag.START.value[1]).type_as(src.data)
    hypotheses = [Hypothesis(value=[str(Tag.START.value[0])], score=None)]

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
        next_word = model.vocab.tgt.get_itos()[next_token]
        if next_word[0] == str(Tag.STOP.value[1]):
            ok = False

        if ok:
            hypotheses[0].value.append(next_word)

        # concatenate word to tensor
        estimation = torch.cat([estimation, torch.zeros(1, 1).type_as(src.data).fill_(next_token)], dim=1)

    hypotheses[0].value.append(str(Tag.STOP.value[1]))

    clean_value = []
    for val in hypotheses[0].value:
        clean_value.append(val)
        if val == Tag.STOP.value[0]:
            break

    hypotheses[0].value.clear()
    for val in clean_value:
        hypotheses[0].value.append(val)

    return hypotheses[0], estimation


"""
Original source :
CS224N 2019-20: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

def beam_search(model, data, beam_size, max_decoding_time_step):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model : NMT Model
    @param data : List of sentences (words) in source language, from test set.
    @param beam_size : beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step : maximum sentence length that Beam search can produce
    @returns hypotheses : List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    # adapt batch list to one sentence case
    sentences = [data]
    with torch.no_grad():

        # decode a test set
        for src_sent in sentences:

            # for each sentences
            h, e = model_beam_search(
                model,
                src_sent,
                beam_size=beam_size,
                max_decoding_time_step=max_decoding_time_step)

    # update model status
    if was_training:
        model.train(was_training)

    return h, e


def model_beam_search(model, batch_entry, beam_size: int = 5, max_decoding_time_step: int = 70):
    """ Given a single source sentence,
    perform beam search,
    yielding translations in the target language.

    @param model                    : learned model
    @param batch_entry              : a single source sentence (words : src, tgt)
    @param beam_size                : beam size
    @param max_decoding_time_step   : maximum number of time steps to unroll the decoding RNN

    @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
    """

    # the memory, encoder output token tensor : (k_t x token_number) and her mask
    src = model.encode(batch_entry.src, batch_entry.src_mask)
    src_mask = batch_entry.src_mask

    # Initialize the hypothesis containers
    hypotheses = [[Tag.START.value[0]]]
    hyp_scores = torch.ones(len(hypotheses), dtype=torch.float, device=model.device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1

        # retrieve tokens from hypothesis
        tgt = torch.tensor([model.vocab.embed_tgt(hyp) for hyp in hypotheses], dtype=torch.long, device=model.device)

        # compute the decoder output based on the attention time_t
        log_p_t = model.step(
            memory=src.expand(tgt.size(0), src.size(1), src.size(2)),
            memory_mask=src_mask.expand(tgt.size(0), src_mask.size(1), src_mask.size(2)),
            x=tgt,
            tgt_mask=None
        )

        # Number of hypothesis that not have been constructed yet
        live_hyp_num = beam_size - len(completed_hypotheses)

        # reconstruct an (k_t X target size) matrix from log_p_t
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
            hyp_word = model.vocab.tgt.get_itos()[hyp_word_id]

            # add new hypotheses to previous state in a new variable
            new_hyp_sent = hypotheses[int(prev_hyp_id)] + [hyp_word]

            # complete sentence are classified, else the new word completed the decoder input for next iteration
            if hyp_word == Tag.STOP.value[0]:
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent, score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        # end loop condition when all hypothesis are inferred
        if len(completed_hypotheses) == beam_size:
            break

        # assign hypothesis with the new
        hypotheses = new_hypotheses

        # create tensor from new hypothesis scores
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=model.device)

    # add hypothesis / score in Hypothesis format
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0], score=hyp_scores[0].item()))

    # sort result (reversed 'cause of the log)
    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
    top_hypothesis = completed_hypotheses[0]
    top_estimation = []

    # format the result
    for h in top_hypothesis.value:
        top_estimation.append(model.vocab.tgt.get_itos().index(h))

    return top_hypothesis, torch.unsqueeze(torch.tensor(top_estimation), 0)


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
            example_hypothesis = model_beam_search_seq2seq(
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


def model_beam_search_seq2seq(model, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70):  # -> List[Hypothesis]:
    """ Given a single source sentence, perform beam search, yielding translations in the target language.
    @param model : learned model
    @param src_sent : a single source sentence (words)
    @param beam_size : beam size
    @param max_decoding_time_step : maximum number of time steps to unroll the decoding RNN
    @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
    """
    # an input token tensor : (words_size x 1)
    src_sentences_var = model.vocab.src.to_input_tensor([src_sent], model.device)

    # encoder returns (1 x words_size x 2*embedding) size tensor and tuple of decoding vectors 1 x embedding
    src_encodings, dec_init_vec = model.encode(src_sentences_var, [len(src_sent)])

    # [attention adaptation : multi head / projection]
    # src_encodings_att_linear = model.att_projection(src_encodings)
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

        # expand (create new view) src encoding à (hyp_num x word x 2*embedding)
        exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

        # expand attention to (hyp_num x word x embedding)
        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                       src_encodings_att_linear.size(1),
                                                                       src_encodings_att_linear.size(2))

        # retrieve tokens from hypothesis
        y_tm1 = torch.tensor([model.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=model.device)

        # embed the prediction to a (hyp_num x embeddings) matrix
        y_t_embed = model.model_embeddings.target(y_tm1)

        # concat (hyp num X 256) embedded prediction and (hyp num X 256) attention to a (hyp num x 512) matrix
        x = torch.cat([y_t_embed, att_tm1], dim=-1)

        # Compute one forward step of the LSTM decoder, including the attention computation.
        (h_t, cell_t), att_t, _ = model.step(x,
                                             h_tm1,
                                             exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             enc_masks=None)

        # log probabilities over target words with projection (1, hyp_num) + (hyp_num, 43280)
        log_p_t = F.log_softmax(model.target_vocab_projection(att_t), dim=-1)

        # Number of hypothesis that need to be constructed
        live_hyp_num = beam_size - len(completed_hypotheses)

        # reconstruct an (hyp_num X target size) matrix from log_p_t
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
