import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
import copy
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from transformers import MambaConfig, MambaModel, GPT2Config, GPT2Model
#from GMM import FineGrainedWeakSupervisor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # fixed
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

        
class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout,batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()


        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        enc_att, att_weight = self._mha_block(self_att_tgt,
                                               memory, memory_mask,
                                               memory_key_padding_mask)
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        
        return x, att_weight
        #return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0] #取0是因为该函数会输出一个包含两元素的元组，这里只取第一个元素
        return self.dropout1(x)
 
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),  att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class StackTransformer(nn.Module):
    r"""StackTransformer is a stack of N decoder layers

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        last_att_weight = None # 【新增】记录注意力权重

        for mod in self.layers:
            output, last_att_weight = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, last_att_weight


class TransformerDecoder(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, embed_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(TransformerDecoder, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=", n_layers)

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding
        # Transformer layer
        decoder_layer = TransformerDecoderLayer(embed_dim, n_head, dim_feedforward=embed_dim * 4,
                                                         dropout=self.dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)

        self.position_encoding = PositionalEncoding(embed_dim, max_len=max_lengths)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(embed_dim, vocab_size)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, encoded_captions, caption_lengths):
        """
        :param x: encoded images, a tensor of dimension (h*w, batch, feature_dim*8)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """  
        word_length = encoded_captions.size(1)
        # triu表示取矩阵的上三角部分（triangular upper），diagonal=1表示从第1条上对角线开始（即主对角线以上的部分），主对角线及以下全置0。
        mask = torch.triu(torch.ones(word_length, word_length), diagonal=1).to(torch.bool)
        mask = mask.to(device)
        tgt_pad_mask = (encoded_captions == self.word_vocab['<PAD>'])

        word_emb = self.vocab_embedding(encoded_captions) #(batch, max_length, embed_dim)
        word_emb = self.position_encoding(word_emb)  # (batch, max_length, embed_dim)

        pred, att_weight = self.transformer(word_emb, x, tgt_mask=mask,
                                    tgt_key_padding_mask=tgt_pad_mask,
                                   )  # (batch, max_length, embed_dim)
        pred = self.wdc(pred)  # (batch, max_length, vocab_size)

        # Sort input data by decreasing lengths,句子长度从大到小
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
            
        decode_lengths = (caption_lengths - 1).tolist()
        #encoded_caption = torch.cat((encoded_captions, torch.zeros([batch, 1], dtype = int).cuda()), dim=1)
        #decode_lengths = (caption_lengths).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind
        
    def sample(self, x, k=1):
        """
        :param x: encoded images, a tensor of dimension (batch_size, channel, enc_image_size* enc_image_size)
        """
        batch = x.size(0)
        
        tgt = torch.zeros(batch, self.max_lengths).to(torch.int64).to(device) #(batch_size, self.max_lengths)

        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths), diagonal=1).to(torch.bool)
        mask = mask.to(device)
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch).to(device) #(batch_size, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch).to(device) #(batch_size, 1)
        #Weight = torch.zeros(1, self.max_lengths, x.size(0)).cuda()
        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<PAD>'])
            word_emb = self.vocab_embedding(tgt) #（batch, self.max_lens, embed_dim) 
            word_emb = self.position_encoding(word_emb) 
            pred, att_weight = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            scores = self.wdc(pred)  # (batch, max_length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim = -1) #(batch,++1)
            #Weight = torch.cat([Weight, weight], dim = 0)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step<(self.max_lengths-1):#except <END> node
                tgt[:, step+1] = predicted_id
        seqs = seqs.squeeze(0) #去掉batch维度，因为batch维度为1，只有一个样本
        seqs = seqs.tolist()
        
        #feature=x.clone()
        #Weight1=Weight.clone()
        return seqs


    def sample_beam(self, x, k=1):
        """
        :param x: encoded images, a tensor of dimension (batch_size, channel, enc_image_size*enc_image_size)
        :param max_lengths: maximum length of the generated captions
        :param k: beam_size
        """
        #batch, channel = x.size(0), x.size(1)
        #L = x.size(2) * x.size(3)
        batch, L, channel = x.size(0), x.size(1), x.size(2) 
        #print(x.shape)
        #print(gmm_memory.shape)
        assert batch == 1, "batch size must be 1"
        x = x.unsqueeze(0).expand(k, -1, -1, -1).reshape(batch*k, L, channel) #(k*batch, h*w, channel)
        tgt = torch.zeros(k*batch, self.max_lengths).to(torch.int64).to(device) #(batch_size*k, self.max_lengths)
        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths), diagonal=1).to(torch.bool)
        #mask = (torch.triu(torch.ones(self.max_lengths, self.max_lengths)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch*k).to(device) #(batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch*k).to(device)
        top_k_scores = torch.zeros(k*batch, 1).to(device)
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<PAD>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = self.position_encoding(word_emb)
            pred, att_weight = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
            scores = self.wdc(pred)  # (batch*k, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch*k, 1, vocab_size] -> [batch*k, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores #(batch*k, vocab_size)
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            # prev_word_inds = top_k_words // vocab_size  # (s)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1) #(k,++1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[prev_word_inds[incomplete_inds], :]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step<self.max_lengths-1:
                tgt[:, :step+2] = seqs


        if complete_seqs == []:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq


    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = fine_tune

