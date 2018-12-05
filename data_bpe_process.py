# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:26:07 2018

@author: mingyang.wang
"""

from bpemb import BPEmb

src_seq_path = 'E:\\data\\machine_translate_ziyan\\train_src.txt'
tgt_seq_path = 'E:\\data\\machine_translate_ziyan\\train_tgt.txt'
src_idx_path = 'E:\\data\\machine_translate_ziyan\\train_src_idx.txt'
tgt_idx_path = 'E:\\data\\machine_translate_ziyan\\train_tgt_idx.txt'

#bpemb_zh = BPEmb(lang='zh', vs=50000, dim=300)

def seq_enc_idx(seq_path, idx_path, lang):
    bpemb_zh = BPEmb(lang=lang, vs=50000, dim=300)
    with open(seq_path, encoding='utf-8') as f1, open(idx_path, 'w', encoding='utf-8') as f2:
        #lines = f1.readlines()
        for line in f1:
            idx = bpemb_zh.encode_ids_with_bos_eos(line.strip())
            f2.write(' '.join([str(i) for i in idx]) + '\n')

#seq_enc_idx(src_seq_path, src_idx_path, lang='en')
#seq_enc_idx(tgt_seq_path, tgt_idx_path, lang='zh')
            

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

pred_path = 'E:/data/NMT_seq_seq_jingsong/test_10/predict_output.txt'
tgt_path = 'E:/data/NMT_seq_seq_jingsong/test_10/translate_corpus/youdao_ch.txt'
print(bleu_score(pred_path, tgt_path))