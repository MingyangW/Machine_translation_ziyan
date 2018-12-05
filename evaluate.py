# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:33:23 2018

@author: mingyang.wang
"""
            

from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk.translate.bleu_score import SmoothingFunction as sf

def bleu_score(pred_path, tgt_path):
    with open(pred_path, encoding='utf-8') as f1, open(tgt_path, encoding='utf-8') as f2:
        pred_seq = f1.readlines()
        tgt_seq = f2.readlines()
        bleu_score=0
        sfunc=sf()
        for i in range(len(pred_seq)):
            bleu_score+=bleu(references=[list(tgt_seq[i])],
                             hypothesis=list(pred_seq[i]),
                         weights=[0.25,0.25,0.25,0.25],
                         smoothing_function=sfunc.method1)
        bleu_score/=len(pred_seq)
    return bleu_score
