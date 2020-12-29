import torch
from torch import nn
import torch.nn.functional as F

from fast_transformer_decoder import FastTransformerDecoder
from transformer_helpers import (
  TokenEmbedding,
  PositionalEncoding,
  weights_init
)

class MusicPerformer(nn.Module):
  def __init__(self, n_token, n_layer, n_head, d_model, d_ff, d_embed,
    activation='relu', dropout=0.1
  ):
    super(MusicPerformer, self).__init__()
    self.n_token = n_token
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation

    self.token_emb = TokenEmbedding(n_token, d_embed, d_model)
    self.d_embed = d_embed

    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(d_model, n_token)

    self.transformer_decoder = FastTransformerDecoder(
      n_layer, n_head, d_model, d_ff, dropout, activation
    )

    self.emb_dropout = nn.Dropout(self.dropout)
    self.apply(weights_init)

  def forward(self, x):
    x_emb = self.token_emb(x)
    x_inp = self.emb_dropout(x_emb) + self.pe(x.size(1)).permute(1, 0, 2)

    dec_out = self.transformer_decoder(x_inp)
    dec_logits = self.dec_out_proj(dec_out)

    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction='mean'
    ).float()

    return {
      'recons_loss': recons_loss,
      'total_loss': recons_loss
    }