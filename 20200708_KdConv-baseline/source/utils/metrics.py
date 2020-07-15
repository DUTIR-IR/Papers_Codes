#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/metrics.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
# from nltk.translate import bleu_score
# from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def accuracy(logits, targets, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    _, preds = logits.max(dim=2)
    trues = (preds == targets).float()
    if padding_idx is not None:
        weights = targets.ne(padding_idx).float()
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    acc = acc.mean()
    return acc


def attn_accuracy(logits, targets):
    """
    logits: (batch_size, vocab_size)
    targets: (batch_size)
    """
    _, preds = logits.squeeze(1).max(dim=-1)
    trues = (preds == targets).float()
    acc = trues.mean()
    return acc


def perplexity(logits, targets, weight=None, padding_idx=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1))
        weight[padding_idx] = 0
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
        nll = nll / word_cnt
    ppl = nll.exp()
    return ppl


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction(epsilon=1e-12).method1,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction(epsilon=1e-12).method1,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):
    """
    distinct
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def cosine(X, Y):
    """
    cosine
    """
    sim = np.sum(X * Y, axis=1) / \
        (np.sqrt((np.sum(X * X, axis=1) * np.sum(Y * Y, axis=1))) + 1e-10)
    return sim


def eval_bleu(ref_resp, hyps_resp):
    """
    compute corpus BLEU score for the hyps_resp
    :param ref_resp: list, a list of reference list
    :param hyps_resp: list, a list of hyps list
    :return: average BLEU score for 1, 2, 3, 4-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_bleu get empty hyps_resp")
        return

    if len(hyps_resp) != len(ref_resp):
        print("ERROR, eval_bleu get un-match hyps_resp %d, and ref_resp %d" % (len(hyps_resp), len(ref_resp)))
        return

    if type(hyps_resp[0]) != str:
        print("ERROR, eval_distinct takes in a list of str, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    return corpus_bleu(ref_resp, hyps_resp, smoothing_function=SmoothingFunction().method1)


def eval_distinct(hyps_resp, n=2):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: distinct score for n-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != str:
        print("ERROR, eval_distinct takes in a list of str, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    num = 0
    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(resp[i: i + n])
            num += 1
    return len(ngram) / num
