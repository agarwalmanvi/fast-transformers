import torch
from torch import nn
import torch.nn.functional as F

device = "cuda:0"

from fast_transformers.builders import AttentionBuilder
from fast_transformers.transformers import TransformerEncoderLayer
from fast_transformers.masking import TriangularCausalMask, LengthMask
from fast_transformers.attention import CausalLinearAttention, AttentionLayer
from fast_transformers.feature_maps import Favor

class FastTransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation='relu'):
    super(FastTransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.dropout = dropout
    self.activation = activation

    att_builder = AttentionBuilder.from_kwargs(
      query_dimensions=d_model // n_head,
      feature_map=Favor.factory(n_dims=d_model // n_head)
    )

    self.attention_layers = [
        AttentionLayer(att_builder.get("causal-linear"), d_model, n_head)
        for l in range(n_layer)
    ]

    self.decoder_layers = nn.ModuleList()
    for l in range(n_layer):
      self.decoder_layers.append(
        TransformerEncoderLayer(
          attention=self.attention_layers[l],
          d_model=d_model,
          d_ff=d_ff,
          dropout=dropout,
          activation=activation
        )
      )

  def forward(self, x, lengths=None, attn_kwargs=None):
    attn_mask = TriangularCausalMask(x.size(1), device=device)

    if lengths is not None:
      length_mask = LengthMask(lengths, device=device)
    else:
      length_mask = None

    # print ('[in decoder]', seg_emb.size(), x.size())

    out = x
    for l in range(self.n_layer):
      # print (out.size())
      out = self.decoder_layers[l](
        out,
        attn_mask=attn_mask,
        length_mask=length_mask,
        attn_kwargs=attn_kwargs
      )

    return out


if __name__ == "__main__":
  dec = FastTransformerDecoder(
    12, 8, 512, 2048, 64
  ).to(device)

  for i in range(1000):
    dec_inp = torch.randn(1, 10240, 512).to(device)
    dec_seg = torch.randn(1, 10240, 64).to(device)
    out = dec(dec_inp, dec_seg)
    print (out.size())
    out.mean().backward()