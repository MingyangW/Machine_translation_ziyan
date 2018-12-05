# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:20:55 2018

@author: mingyang.wang
"""

import argparse
import torch
import Constant
from bpemb import BPEmb
#from torch.utils.data import Dataset, TensorDataset, DataLoader

 
def split_seq_pairs(seq_pairs_path, src_seq_path, tgt_seq_path, encoding='utf-8', sep='\t'):
    """读取双语料文件，返回单语料文件"""
    with open(seq_pairs_path, encoding=encoding) as f:
        lines = f.readlines()
        pairs = [line.strip().split(sep) for line in lines if line.strip()]
        src_seq = [pair[0] for pair in pairs]
        tgt_seq = [pair[1] for pair in pairs]
        with open(src_seq_path, 'w', encoding=encoding) as src:
            src.write('\n'.join(src_seq))
        with open(tgt_seq_path, 'w', encoding=encoding) as tgt:
            tgt.write('\n'.join(tgt_seq))
"""            
def seqs_2_word(src_seq_path, tgt_seq_path, encoding, src_lang_type='eng', tgt_lang_type='ch'):
    # --加载源语料与目标语料,返回分词后转化为list格式的语料
    src_seq2words = seq_2_word(src_seq_path, lang_type=src_lang_type)
    tgt_seq2words = seq_2_word(tgt_seq_path, lang_type=tgt_lang_type)
    if len(src_seq2words) != len(tgt_seq2words):
        print('<Warning> The train data don’t match the test data, the length is different!')
        return None
    return src_seq2words, tgt_seq2words            
"""
def seq_2_word(seq_path, encoding='utf-8', lang_type='en'):
    """读取单语料文件，返回分词后的list格式"""
    with open(seq_path, encoding=encoding) as f:
        blank_rows = 0
        lines = f.readlines()
        if lang_type == 'en':
            words= [list(line.strip().lower().split(' ')) for line in lines if line.strip()]
        elif lang_type == 'ch':
            words = [list(line.strip()) for line in lines if line.strip()]
        else:
            print('<Warning> Don’t support this language!')
            exit()
        if blank_rows:
            print('<Info> Seq file contain {} blanks rows!'.format(blank_rows))
    return words
                
def seqs_2_index(seqs, word_dict):
    """将语料转化为index形式"""
    seqs_index = [[Constant.BOS]+seq_2_index(seq, word_dict)+[Constant.EOS] for seq in seqs]
    return seqs_index
        
def seq_2_index(seq, word_dict):
    """将语句转化为index形式"""
    return [word_dict[word] if word in word_dict else Constant.UNK for word in seq]

def index_to_tensor(src_index, tgt_index, max_len=100):
    """将index转换成等长tensor"""
    i = 0
    while i < len(src_index):
        if len(src_index[i]) <= max_len and len(tgt_index[i]) <= 100:
            src_index[i] = src_index[i] + (max_len - len(src_index[i])) * [0]
            tgt_index[i] = tgt_index[i] + (max_len - len(tgt_index[i])) * [0]
            i += 1
        else:
            src_index.pop(i)
            tgt_index.pop(i)
    src_tensor = torch.tensor(src_index)
    tgt_tensor = torch.tensor(tgt_index)
    return src_tensor, tgt_tensor
    

def pretrain_bpe_seq_2_index(seq_path, encoding='utf-8', lang_type='en'):
    """读取单语料文件，返回分词后index的list格式创建词典，"""
    with open(seq_path, encoding=encoding) as f:
        lines = f.readlines()
        if lang_type == 'en':
            bpemb_en = BPEmb(lang='en', vs=50000, dim=300)
            indexs = [bpemb_en.encode_ids_with_bos_eos(line.strip().lower()) \
                     for line in lines if line.strip()]
            words = ['<blank>'] + bpemb_en.words
        elif lang_type == 'ch':
            bpemb_zh = BPEmb(lang='zh', vs=50000, dim=300)
            indexs = [bpemb_zh.encode_ids_with_bos_eos(line.strip()) \
                     for line in lines if line.strip()]
            words = ['<blank>'] + bpemb_zh.words
        else:
            print('<Warning> Don’t support this language!')
            exit()
        indexs = [[i+1 for i in index]for index in indexs]
        word_dict = {word:idx for idx,word in enumerate(words)}
    return indexs, word_dict

        
class CreateDict:
    """创建词典"""
    def __init__(self, name):
        self.name = name
        self.word2index = {Constant.PAD_WORD: 0, Constant.UNK_WORD: 1, Constant.BOS_WORD: 2, Constant.EOS_WORD: 3}
        #self.index2word = {0: Constant.PAD_WORD, 1: Constant.UNK_WORD, 2: Constant.BOS_WORD, 3 :Constant.EOS_WORD}
        self.word_count = {}
        self.n_words = 4 # Count PAD、UNK、BOS、EOS

    def add_seq_path(self, seq_path, encoding='utf-8', lang_type='eng'):
        """通过语料文件创建字典"""
        with open(seq_path, encoding=encoding) as f:
            lines = f.readlines()
            for line in lines:
                self.add_seq(line.strip())
    
    def add_seq_words(self, seq2words):
        """通过语料词列表创建字典"""
        for seq in seq2words:
            for word in seq:
                if word:
                    self.add_word(word)
    
    def add_seq(self, seq, lang_type='eng'):
        """将语句转化为词"""
        words = seq.split(' ') if lang_type == 'eng' else list(seq)
        for word in words:
            if word:
                self.add_word(word)
            
    def add_word(self, word):
        """将词加入字典"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            #self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
            


def main():         
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-split_seq_pairs', action='store_true')
    parser.add_argument('-seq_pairs_path', type=str, default=None)
    parser.add_argument('-src_seq_path', type=str, default=None)
    parser.add_argument('-tgt_seq_path', type=str, default=None)
    parser.add_argument('-save_data_path', type=str, default=None)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-pretrain_bpe', action='store_true')
    
    opt = parser.parse_args()
    
    work_path = 'E:/data/machine_translate_ziyan/10/'
    opt.seq_pairs_path = work_path + 'data/cmn.txt'
    opt.src_seq_path = work_path + 'data/train_src.txt'
    opt.tgt_seq_path = work_path + 'data/train_tgt.txt'
    opt.save_data_path = work_path +'data/data_processed.pth'
    opt.pretrain_bpe = True
    
    
    # --将双语料数据对文本转化为单语料文本
    #split_seq_pairs(opt.seq_pairs_path, opt.src_seq_path, opt.tgt_seq_path) 
    
    if opt.pretrain_bpe:
        # --将语料文本切分转化为（词、字）的list列表,创建词典
        src_seq2idx, src_word2index = pretrain_bpe_seq_2_index(opt.src_seq_path, lang_type='en') 
        tgt_seq2idx, tgt_word2index = pretrain_bpe_seq_2_index(opt.tgt_seq_path, lang_type='ch')
        # --创建源语言与目标语言的词典
        src_dict, tgt_dict =  CreateDict('src_lang'), CreateDict('tgt_lang')
        src_dict.word2index = src_word2index
        tgt_dict.word2index = tgt_word2index

    else:
        # --将语料文本切分转化为（词、字）的list列表
        src_seq2words = seq_2_word(opt.src_seq_path, lang_type='en') 
        tgt_seq2words = seq_2_word(opt.tgt_seq_path, lang_type='ch')
        
        # --比较源语料与目标语料长度是否一致
        if len(src_seq2words) != len(tgt_seq2words):
            print('<Warning> The sequence number is different between src and tgt!')
            exit()
            
        # --删除超出最大长度的语料对
        beyond_seq = 0
        for i in range(len(src_seq2words)):
            if len(src_seq2words[i]) > opt.max_seq_len or len(tgt_seq2words[i]) > opt.max_seq_len:
                src_seq2words.pop(i) 
                tgt_seq2words.pop(i)
                beyond_seq += 1
        if beyond_seq:
            print('<Info> {} sequences beyond the maximum length!')
    
        # --创建源语言与目标语言的词典
        src_dict, tgt_dict =  CreateDict('src_lang'), CreateDict('tgt_lang')
        src_dict.add_seq_words(src_seq2words)
        tgt_dict.add_seq_words(tgt_seq2words)
    
        # --将语料的语句转成index形式
        src_seq2idx = seqs_2_index(src_seq2words, src_dict.word2index)
        tgt_seq2idx = seqs_2_index(tgt_seq2words, tgt_dict.word2index)
    

    # -- 保存数据
    data = {
            'setting':opt,
            'dict':{'src':src_dict.word2index,
                    'tgt':tgt_dict.word2index},
            'train':{'src':src_seq2idx,
                     'tgt':tgt_seq2idx}
            }
    
    torch.save(data, opt.save_data_path)
    print('<Info> Data preprocess finish!')

if __name__ == '__main__':
    
    main()
    # --将index语料转成tensor
    #src_tensor, tgt_tensor = data1.index_to_tensor(src_seq_idx, tgt_seq_idx)
    #print(src_tensor.size(), tgt_tensor.size())
    
    # --创建生成器，生成训练数据
    #dataset = TensorDataset(src_tensor, tgt_tensor)
    #dataiter = DataLoader(dataset, batch_size=2)

