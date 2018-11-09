# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:47:31 2018

@author: mingyang.wang
"""

import preparedata
import random
import time
import math
import torch
import torch.nn.functional as F
from torch import optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD )#reduction='sum'

    return loss


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1) #####


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train_epoch(model, optimizer, pairs, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(7500)]
    for i in range(1, 10):
        training_pair = training_pairs[i - 1]
        input_length = training_pair[0].size(1)
        target_length = training_pair[1].size(1)
        
        input_length = torch.LongTensor([[input_length]])
        target_length = torch.LongTensor([[target_length]])

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = training_pair[0],input_length, training_pair[1],target_length
        gold = tgt_seq[:, 1:]


        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, optimizer, pairs, device):
    
    for epoch_i in range(10):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, optimizer, pairs, device, smoothing=False)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
input_lang, output_lang, pairs = preparedata.prepareData('eng', 'fra', True)

transformer = Transformer(
    input_lang.n_words,
    output_lang.n_words,
    MAX_LENGTH,
    tgt_emb_prj_weight_sharing=False,
    emb_src_tgt_weight_sharing=False).to(device)

optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=1e-09), 512, 4000)

train(transformer, optimizer, pairs, device)

