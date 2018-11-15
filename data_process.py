# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:20:55 2018

@author: mingyang.wang
"""

import torch
import Constant
from torch.utils.data import TensorDataset, DataLoader



class LoadProcess():
    """加载、处理单语料数据"""
    def load_seq(self, file_path, encoding='utf-8', lang_type='eng'):
        """读取文件，返回切分后的list"""
        try:
            with open(file_path, encoding=encoding) as f:
                lines = f.readlines()
                if lang_type == 'eng':
                    data = [list(line.strip().lower().split(' ')) for line in lines]
                elif lang_type == 'ch':
                    data = [list(line.strip()) for line in lines]
                else:
                    print('Don’t support this language!')
        except:
            print('Load sequence error!')
        return data, len(lines)
        
    def data_preprocess(self, src_path, tgt_path):
        """加载单语料,转化为list"""
        try:
            src_seq, len_src_seq = self.load_seq(src_path, lang_type='eng')
            tgt_seq, len_tgt_seq = self.load_seq(tgt_path, lang_type='ch')
            if len_src_seq != len_tgt_seq:
                print('The train data don’t match the test data, the length is different!')
                return None
            return src_seq, len_src_seq, tgt_seq, len_tgt_seq
        except:
            print('data preprocess error!')
    
    def split_data_parirs(self, data_path, src_path, tgt_path, encoding='utf-8', sep='\t'):
        """将双语料文件处理为单语料"""
        try:
            with open(data_path, encoding=encoding) as f:
                lines = f.readlines()
                datas = [line.strip().split(sep) for line in lines]
                src_data = [data[0] for data in datas]
                tgt_data = [data[1] for data in datas]
                with open(src_path, 'w', encoding=encoding) as src:
                    src.write('\n'.join(src_data))
                with open(tgt_path, 'w', encoding=encoding) as tgt:
                    tgt.write('\n'.join(tgt_data))
        except:
            print('split_data_pairs error!')
                    
    def seq_to_index(self, data, data_dict):
        """将语料转化为index形式"""
        data_index = [[BOS]+self.word_to_index(seq, data_dict)+[EOS] for seq in data]
        return data_index
            
    def word_to_index(self, seq, data_dict):
        """将语句转化为index形式"""
        return [data_dict[word] if word in data_dict else UNK for word in seq]
    
    def index_to_tensor(self, src_index, tgt_index, max_len=100):
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
    
        
class createDict:
    """创建词典"""
    def __init__(self, name):
        self.name = name
        self.word2index = {PAD_WORD: 0, UNK_WORD: 1, BOS_WORD: 2, EOS_WORD: 3}
        self.index2word = {0: PAD_WORD, 1: UNK_WORD, 2: BOS_WORD, 3 :EOS_WORD}
        self.word_count = {}
        self.n_words = 4 # Count PAD、UNK、BOS、EOS

    def add_data(self, data_path, encoding='utf-8', lang_type='eng'):
        with open(data_path, encoding=encoding) as f:
            lines = f.readlines()
            for line in lines:
                self.add_sentence(line.strip())
    
    def add_sentence(self, sentence, lang_type='eng',):
        if lang_type == 'eng':
            for word in sentence.split(' '):
                self.add_word(word)
        elif lang_type == 'ch':
            for word in list(sentence):
                self.add_word(word)
            
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
            

    
def load_dict(dict_path=None, src_path=None, tgt_path=None, create=True):
    """加载或创建字典"""
    if create:
        src_dict = createDict('src')
        tgt_dict = createDict('tgt')
        src_dict.add_data(src_path)
        tgt_dict.add_data(tgt_path, lang_type='ch')
        data_dict = {'src':src_dict, 'tgt':tgt_dict}
        with open(dict_path, 'wb') as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
        return src_dict, tgt_dict
    else:
        with open(dict_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict['src'], data_dict['tgt']
    
def save_data(src_dict, tgt_dict, src_seq_idx, tgt_seq_idx, data_path):
    data_dict = {
            'src_dict':src_dict.word2index,
            'tgt_dict':tgt_dict.word2index,
            'src_seq_idx':src_seq_idx,
            'tgt_seq_idx':tgt_seq_idx
            }
    torch.save(data_dict, data_path)
    
#    with open(data_path, 'wb') as f:
#            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


        
file_path = 'E:\\data\\translate\\cmn-eng\\1112.txt'
data_path = 'E:\\data\\translate\\1112.txt'
src_path = 'E:\\data\\machine_translate\\valid_src.txt'
tgt_path = 'E:\\data\\machine_translate\\valid_tgt.txt'
dict_path = 'E:\\data\\translate\\1112.pkl'
all_data_path = 'E:\\data\\translate\\data1113.pth'
max_len = 100

data1 = LoadProcess()
# --将双语料数据对文本转化为单语料文本
data1.split_data_parirs(file_path,src_path, tgt_path) 
#print(data1.data_preprocess(src_path, tgt_path))
# --将语料文本切分转化为（词、字）的list列表
src_seq, len_src_seq, tgt_seq, len_tgt_seq = data1.data_preprocess(src_path, tgt_path) 
# --创建并加载词典
src_dict, tgt_dict = load_dict(dict_path=dict_path, src_path=src_path, tgt_path=tgt_path) 
# --加载词典
#src_dict, tgt_dict = load_dict(dict_path=dict_path, create=False) 
# --将语料的语句转成index形式
src_seq_idx = data1.seq_to_index(src_seq, src_dict.word2index)
tgt_seq_idx = data1.seq_to_index(tgt_seq, tgt_dict.word2index)

# -- 将字典、语料保存
save_data(src_dict, tgt_dict, src_seq_idx, tgt_seq_idx, all_data_path)
# --将index语料转成tensor
src_tensor, tgt_tensor = data1.index_to_tensor(src_seq_idx, tgt_seq_idx)
print(src_tensor.size(), tgt_tensor.size())

# --创建生成器，生成训练数据
dataset = TensorDataset(src_tensor, tgt_tensor)
dataiter = DataLoader(dataset, batch_size=2)




#for i in dataiter:
#    print(i[0].size())
#    print(i[1].size())
#    break

#print(src_dict.n_words, tgt_dict.n_words)   
#print(src_dict.word2index)
#print(src_seq_idx)
#print(tgt_seq_idx)
