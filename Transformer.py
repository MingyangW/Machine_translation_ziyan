# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:41:24 2018

@author: mingyang.wang
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """缩放·点积·注意力机制"""
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, q, k, v, scale=None, atten_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1,2))
        if scale:
            attention *= scale
        if atten_mask is not None:
            #print('attention:',attention.size())
            #print('atten_mask', atten_mask.size())
            attention = attention.masked_fill(atten_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm  = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, atten_mask=None):
        residual = q  # --残差连接
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = k.size(0)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(batch_size*num_heads, -1, dim_per_head)
        k = k.view(batch_size*num_heads, -1, dim_per_head)
        v = v.view(batch_size*num_heads, -1, dim_per_head)

        if atten_mask is not None:
            atten_mask = atten_mask.repeat(num_heads, 1, 1)
        scale = (k.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(q, k, v, scale, atten_mask)

        context = context.view(batch_size, -1, dim_per_head*num_heads)

        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual+output)

        return output, attention

"""
decoder的self-attention，里面使用到的scaled dot-product attention，
同时需要padding mask和sequence mask相加作为attn_mask。
其他情况，attn_mask一律等于padding mask。
"""
def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask
 
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        position_encoding = np.array([[pos/(10000**(2*(i//2)/d_model)) 
        for i in range(d_model)]for pos in range(max_seq_len)])
        
        position_encoding[:,0::2] = np.sin(position_encoding[:,0::2])
        position_encoding[:,1::2] = np.cos(position_encoding[:,1::2])
        
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.Tensor(position_encoding)
        position_encoding = torch.cat((pad_row, position_encoding))
        
        self.position_enccoding = nn.Embedding(max_seq_len+1, d_model)
        self.position_enccoding.weight = nn.Parameter(position_encoding, requires_grad=False)
        
    def forward(self, input_len):
        """
        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        max_len = max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        
        input_pos = tensor([list(range(1, i + 1)) + [0] * (max_len - i).item() for i in input_len])
        return self.position_enccoding(input_pos)


class PositionalWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward network"""
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, x):
        output = x.transpose(1,2)
        output = self.w2(F.relu(self.w1(output)))
        output = output.transpose(1,2)
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        return output
            
    
class Encoderlayer(nn.Module):
    """Encoderlayer"""
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoderlayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        
    def forward(self, inputs, atten_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, atten_mask)
        output = self.feed_forward(context)
        return output, attention
    
class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, 
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8, 
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
                [Encoderlayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    
    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        
        self_attention_mask = padding_mask(inputs, inputs)
        
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        
        return output, attentions


class DecoderLayer(nn.Module):
    """DecoderLayer"""
    def __init__(self,model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        
    def forward(self, dec_inputs, enc_inputs, self_atten_mask=None, context_atten_mask=None):
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs,
                                                    dec_inputs, self_atten_mask)
        #dec_output, context_attention = self.attention(enc_inputs, enc_inputs, 
        #                                               dec_inputs, context_atten_mask)
        dec_output, context_attention = self.attention(dec_inputs, enc_inputs, 
                                                       enc_inputs, context_atten_mask)
        
        dec_output = self.feed_forward(dec_output)
        return dec_output, self_attention, context_attention
    
class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, 
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers 
        self.decoder_layers = nn.ModuleList(
                [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        
    def forward(self, inputs, inputs_len, enc_output, context_atten_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        #a = self.pos_embedding(inputs_len)
        #print('a:',a.size())
        
        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_atten_mask = torch.gt((self_attention_padding_mask+seq_mask), 0)
        
        self_attentions = []
        context_attentions = []
        
        for decoder in self.decoder_layers:
            output, self_atten, context_atten = decoder(
                    output, enc_output, self_atten_mask, context_atten_mask)
            self_attentions.append(self_atten)
            context_attentions.append(context_atten)
            
        return output, self_attentions, context_attentions
    

class Transformer(nn.Module):
    """Transformer"""
    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, 
                               model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers,
                               model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_atten_mask = padding_mask(src_seq, tgt_seq)
        #print('src_seq',src_seq.size())
        #print('tgt_seq',tgt_seq.size())
        #print('context_atten_mask',context_atten_mask.size())
        output, enc_self_atten = self.encoder(src_seq, src_len)
        output, dec_self_atten, ctx_atten = self.decoder(tgt_seq, tgt_len, output, context_atten_mask)
        
        output = self.linear(output)
        output = self.softmax(output)
        output = output.view(-1, output.size(2))
        return output, enc_self_atten, dec_self_atten, ctx_atten