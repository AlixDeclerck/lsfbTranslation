"""
Original source :
CS224N 2019-20: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import torch
from typing import List
import torch.nn.functional as F
from common.constant import Hypothesis


def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70):  # -> List[Hypothesis]:
    """ Given a single source sentence, perform beam search, yielding translations in the target language.
    @param src_sent (List[str]): a single source sentence (words)
    @param beam_size (int): beam size
    @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
    @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
    """
    src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

    src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])

    # [adapt attention]
    # src_encodings_att_linear = self.att_projection(src_encodings)
    src_encodings_att_linear = self.att_multiHeaded(src_encodings)

    h_tm1 = dec_init_vec
    att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

    eos_id = self.vocab.tgt['</s>']

    hypotheses = [['<s>']]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
    completed_hypotheses = []

    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1
        hyp_num = len(hypotheses)

        exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                       src_encodings_att_linear.size(1),
                                                                       src_encodings_att_linear.size(2))

        y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
        y_t_embed = self.model_embeddings.target(y_tm1)

        x = torch.cat([y_t_embed, att_tm1], dim=-1)

        (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

        # log probabilities over target words
        log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

        live_hyp_num = beam_size - len(completed_hypotheses)
        continuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(continuating_hyp_scores, k=live_hyp_num)

        prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
        hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = self.vocab.tgt.id2word[hyp_word_id]
            new_hyp_sent = hypotheses[int(prev_hyp_id)] + [hyp_word]
            if hyp_word == '</s>':
                completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                       score=cand_new_hyp_score))
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(completed_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
        h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
        att_tm1 = att_t[live_hyp_ids]

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                               score=hyp_scores[0].item()))

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses
