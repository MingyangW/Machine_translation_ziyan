# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:52:43 2018

@author: mingyang.wang
"""

import re
import torch
import random
import argparse
import unicodedata
import numpy as np
from io import open

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# --语料标准化，小写、删除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# --限制语料长度
def filterPair(p, max_len):
    return len(p[0].split(' ')) < max_len and \
        len(p[1].split(' ')) < max_len and \
        p[0].startswith(eng_prefixes)
        
def filterPair1(p, max_len):
    return len(p[0].split(' ')) < max_len and \
        len(p[1].split(' ')) < max_len
    

def indexFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return indexes

# --语料转成index形式
def indexFromPair(src_lang, tgt_lang, pairs):
    src_idx, tgt_idx = [], []
    for pair in pairs:
        src_idx.append(indexFromSentence(src_lang, pair[0]))
        tgt_idx.append(indexFromSentence(tgt_lang, pair[1]))
    print('src_idx:',src_idx)
    return src_idx, tgt_idx


# --读取数据，txt文件，形如“你好 hello”的语料，\t语种分隔,\n语句对分隔
def prepareData(opt):
    print("Reading lines...")
    
    lines = open(opt.path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
    if opt.reverse:
        pairs = [list(reversed(p)) for p in pairs]
    src_lang, tgt_lang = Lang(opt.src_lang), Lang(opt.tgt_lang)

    print("Read %s sentence pairs" % len(pairs))
    pairs = [pair for pair in pairs if filterPair(pair, opt.max_len)]
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    
    for pair in pairs:
        src_lang.addSentence(pair[0])
        tgt_lang.addSentence(pair[1])
        
    print("Counted words:")
    print(src_lang.name, src_lang.n_words)
    print(tgt_lang.name, tgt_lang.n_words)
    src_idx, tgt_idx = indexFromPair(src_lang, tgt_lang, pairs)
    data = {
            'setting': opt,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'pairs': pairs,
            'src_idx': src_idx,
            'tgt_idx': tgt_idx
            }
    torch.save(data, opt.data_save_path)
    print('Process data over!')
    print('Random choice a pair:\n',random.choice(pairs))
    
    



#############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='E:\\data\\translate\\data\\data\\eng-fra.txt')
    parser.add_argument('-reverse', action='store_true', default=False)#
    parser.add_argument('-src_lang', type=str, default='source_lang')
    parser.add_argument('-tgt_lang', type=str, default='target_lang')
    parser.add_argument('-max_len', type=int, default=10)
    parser.add_argument('-data_save_path', type=str, default='E:\\data\\translate\\process_data.t7')
    
    opt = parser.parse_args()
    prepareData(opt)


if __name__ == '__main__':
    main()
    
