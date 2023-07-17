from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

    ### TODO
    #Initial dimensions: [bs, num_attention_heads, seq_len, attention_head_size]
    key = torch.transpose(key,0,1)
    query = torch.transpose(query,0,1)
    value = torch.transpose(value,0,1)
    #Change to [num_attention_heads,bs seq_len, attention_head_size]
    key_transposed = torch.transpose(key,2,3)

  
    #Change to [num_attention_heads,bs, attention_head_size, seq_len]
    query_head_list = torch.split(query, 1, dim=0)
    key_transposed_head_list = torch.split(key_transposed, 1, dim=0)
    value_head_list = torch.split(value, 1, dim=0)

    dk = key.shape[3]
    seq_len = key.shape[2]
    #repeated_attention_mask = attention_mask.repeat(seq_len,1,1).transpose(0,1) #This is causing an error. Probably from class Bert the extended attention already adjusts shape
    # Initialize an empty list to store the result
    result_list = []

    # Perform matrix multiplication for each pair of tensors
    for Query, Key, Value in zip(query_head_list, key_transposed_head_list, value_head_list):
       
        result_tensor = torch.matmul(Query, Key)
        # result tensor size [BS,seq_len, seq_len]
        # attention_mask [BS, seq_len]
        
        result_tensor = result_tensor.masked_fill(attention_mask==0,-99999999999)
        result_tensor =result_tensor/math.sqrt(dk)
        
        result_tensor = torch.nn.functional.softmax(result_tensor,dim=-1)
        #[BS,seq_len, seq_len]
        # The sum of the softmax scores in the third dimension sum to 1
        result_tensor = torch.matmul(result_tensor,Value)

        result_list.append(result_tensor)

    concatenated_tensor = torch.cat(result_list, dim=0)
    #[number of heads, BS,seq_len, seq_len]
    print('concatenated tensor shape', concatenated_tensor.shape)
    return concatenated_tensor


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
    
    # Each sub-layer in each encoder has a residual connection around it leading to layer-normalisation
    
    # Need to combine input and output
    print('input shape',input.shape)
    print('output shape',output.shape)
    residual = input + output
    
    # Apply normalisation
    norm_output = ln_layer(residual)
    
    # I am not sure if dropout is best applied at this part of the code but dropout adds regularisation
    dropout_norm_output = dropout(norm_output)
    
    # Introduce non-linearity with dense_layer
    transformed_norm_output = dense_layer(dropout_norm_output)
    
    
    return transformed_norm_output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer 
    4. a add-norm that takes the input and output of the feed forward layer
    """
    ### TODO
    self_attention_output = self.self_attention.forward(hidden_states,attention_mask)

    normalized_attention_layer = self.add_norm(input=hidden_states, output=self_attention_output , 
                  dense_layer= self.attention_dense, dropout=self.attention_dropout, ln_layer= self.attention_layer_norm)
    
    ffn = self.interm_dense(normalized_attention_layer)
    ffn = self.interm_af(ffn)
    normalized_output_layer = self.add_norm(input=normalized_attention_layer, output=ffn, 
                  dense_layer= self.out_dense, dropout=self.out_dropout, ln_layer= self.out_layer_norm)
    

    raise normalized_output_layer




class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = self.word_embedding(input_ids)

    # Get position index and position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length] #subsets a list of positions 0:512 to 0:seq_length. 

    pos_embeds = self.pos_embedding(pos_ids)


    # Get token type ids, since we are not consider token type, just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    hidden_states = inputs_embeds+pos_embeds+tk_type_embeds
    hidden_states = self.embed_layer_norm(hidden_states)
    hidden_states = self.embed_dropout(hidden_states)

    return hidden_states


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
