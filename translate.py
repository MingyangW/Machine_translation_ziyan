# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:24:58 2018

@author: mingyang.wang
"""

import os
import torch
import argparse
import torch.utils.data as td
import Data_preprocessing as dp
from tqdm import tqdm
from Translation import Translator
from dataset import TranslationDataset, collate_fn



def main():
    work_path = 'E:/data/machine_translate_ziyan/10/'
    model_path = work_path + 'save_model/model.chkpt'
    input_path = work_path + 'test_input.txt' 
    output_path = work_path + 'test_output.txt'
    vocab_path = work_path + 'data/data_processed.pth'
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument('-model_path', type=str, default=model_path)
    parser.add_argument('-input_path', type=str, default=input_path)
    parser.add_argument('-output_path', type=str, default=output_path)
    parser.add_argument('-vocab_path', type=str, default=vocab_path)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-n_best', type=int, default=1)
    parser.add_argument('-cuda', action='store_true')
    
    opt = parser.parse_args()
    
    data_state_dict = torch.load(opt.vocab_path)
    
    #-- 翻译文件预处理，转化为index形式
    if data_state_dict['setting'].pretrain_bpe:
        input_data_index = dp.pretrain_bpe_seq_2_index(opt.input_path, lang_type='en')
    else:
        input_split_word = dp.seq_2_word(opt.input_pth, lang_type='en')
        input_data_index = dp.seqs_2_index(input_split_word, data_state_dict['dict']['src'])
        
    input_data = td.DataLoader(
            TranslationDataset(
                    src_word2idx=data_state_dict['dict']['src'],
                    tgt_word2idx=data_state_dict['dict']['tgt'],
                    src_insts=input_data_index),
            num_workers=0,
            batch_size=opt.batch_size,
            collate_fn=collate_fn)
            
    translater = Translator(opt)
    for batch in tqdm(input_data, mininterval=2, desc='  - (Translating)   ', leave=False):
        translater.translate_batch(*batch)
        break


if __name__ == '__main__':
    main()