# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:14:01 2020

@author: s6532600
"""

from typing import Iterable, Optional, Tuple
import collections
import torch
from torch import Tensor
from torch.nn import functional as F
import spacy

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
        
def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens        

def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def remove_special_tokens(s, model, tokenizer='decoder'):
    special_tokens = [model.tokenizer[tokenizer].bos_token_id
                      ,model.tokenizer[tokenizer].eos_token_id
                      ,model.tokenizer[tokenizer].pad_token_id
                      ,model.tokenizer[tokenizer].cls_token_id
                      ,model.tokenizer[tokenizer].sep_token_id]
    return [w for w in s if w not in special_tokens]

def tensor_to_tokens(t, model, tokenizer='decoder', rm_special_tokens=False):
    if len(t.size())==1:
        t = t[None,:]
    sentences = [t[i] for i in range(len(t))]
    if rm_special_tokens:
        sentences = [remove_special_tokens(s, model, tokenizer) for s in sentences]
    tokens = [model.tokenizer[tokenizer].convert_ids_to_tokens(s) for s in sentences]
    strings = [model.tokenizer[tokenizer].convert_tokens_to_string(s) for s in tokens]
    return strings[0] if len(strings) == 1 else strings

def review_data(data, model, prediction=None, rm_special_tokens=False):
    review = []
    for i in range(len(data['y']['input_ids'])):
        p = tensor_to_tokens(data['persona_info_h']['input_ids'][i],model,'encoder',rm_special_tokens)
        d = tensor_to_tokens(data['y']['input_ids'][i],model,'decoder',rm_special_tokens)
        review.append((p,d))
        if prediction is not None:
            pred = tensor_to_tokens(prediction[i],model,'decoder',rm_special_tokens)
            review[-1] = review[-1]+(pred,)        
    return review

def move_to_device(data, target):
    
    if isinstance(data, torch.Tensor) and data.device.type != target:
        data_torch = data.to(target)
        
    if isinstance(data, dict):
        data_torch = {}
        for k,v in data.items():
            if isinstance(v, dict):
                data_torch[k] = {}
                for k1,v1 in v.items():
                    data_torch[k][k1] = v1.to(target) if isinstance(v1,torch.Tensor) else v1
            else:
                data_torch[k] = v.to(target) if isinstance(v,torch.Tensor) else v
    
    if (isinstance(data, tuple) or isinstance(data, list)) and len(data)>0:
        data_torch = []
        for d in data:
            data_torch.append({})
            for k, v in d.items():
                data_torch[-1][k] = v.to(target) if isinstance(v,torch.Tensor) else v
                
    return data_torch

def init_spacy(disable=[]):
    print("Loading spacy en_core_web_md...")
    try:
        spacy_parser = spacy.load('en_core_web_md',disable=disable)  # if you're getting an error here, run "python -m spacy download en_core_web_md"
    except:
        spacy_parser = spacy.load(r"C:\Users\s6532600\.conda\envs\YW\lib\site-packages\en_core_web_md\en_core_web_md-2.3.1",disable=disable)
    print("Finished loading spacy en_core_web_md")
    return spacy_parser

def get_spacy_encoded_text(sentences, spacy_parser):
    encoded_sentences = [spacy_parser(sent) for sent in sentences]
    return encoded_sentences

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def convert_special_tokens_to_unk(x, tokenizer):
    special_tokens = [tokenizer.__getattribute__(t) for t in tokenizer.SPECIAL_TOKENS_ATTRIBUTES if t!='unk_token']
    special_tokens = [t for t in special_tokens if isinstance(t,str)]
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    for tid in special_token_ids:
        x[x==tid] = tokenizer.unk_token_id
    return x
        
