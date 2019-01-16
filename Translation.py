# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:44 2018

@author: mingyang.wang
"""

import torch
from Model import Transformer

class Translator():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')
        
        checkpoint = torch.load(opt.model_path)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt
        
        model = Transformer(
                src_vocab_size=model_opt.src_vocab_size,
                src_max_len=model_opt.max_token_seq_len,
                tgt_vocab_size=model_opt.tgt_vocab_size,
                tgt_max_len=model_opt.max_token_seq_len,
                num_layers=model_opt.n_layers,
                model_dim=model_opt.d_model,
                num_heads=model_opt.n_head,
                ffn_dim=model_opt.d_inner_hid,
                dropout=model_opt.dropout)  
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')
        
        model = model.to(self.device)
        self.model = model
        self.model.eval()
        
    def translate_batch(self, src_seq, src_len):
        with torch.no_grad():
            #-- Encoder
            src_seq, src_len = src_seq.to(self.device), src_len.to(self.device)
            src_enc, *_ = self.model.encoder(src_seq, src_len)
            print('\n')
            print('src_seq, src_len:',src_len.size())
            print('src_len:',src_len.size())
            print('src_enc:',src_enc.size())
            