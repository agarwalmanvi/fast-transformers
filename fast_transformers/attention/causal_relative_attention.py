#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the full attention similar to the one implemented by PyTorch's
MultiHeadAttention module. Note that this module is to be used in conjuction
with the `fast_transformers.attention.attention_layer.AttentionLayer` in order
to work."""

from math import sqrt

import torch
from torch.nn import Dropout, Module, Parameter
import torch.nn.functional as F

from ..attention_registry import AttentionRegistry, Optional, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher, AttentionEvent


class CausalRelativeAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, num_heads, max_pos, emb_dim,
                 softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(CausalRelativeAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.pos_emb = Parameter(
            torch.normal(mean=0., std=emb_dim ** -.5,
                         size=(max_pos, num_heads, emb_dim)))

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N, L, H, D = queries.shape
        _, S, _, _ = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(D)

        if L != S:
            raise RuntimeError(("CausalRelativeAttention only supports "
                                "self-attention (L == S)"))

        # Relative positional encodings have shape (R, H, D), where the first
        # dimension corresponds to lags <= 0. Pad it or clip it to L
        E = self.pos_emb[:L]
        E = F.pad(E, (0, 0, 0, 0, L - E.shape[0], 0))

        # Efficient causal relative attention as per Huang et al., 2018
        QE = torch.einsum("nlhd,rhd->nhlr", queries, E)
        # Skewing: pad, reshape and clip
        S = F.pad(QE, (1, 0)).reshape(N, H, L + 1, L)[:, :, 1:, :]

        # Compute the logits, add the relative logits and apply the masks
        QK = torch.einsum("nlhd,nshd->nhls", queries, keys)
        QK = QK + S
        if not attn_mask.lower_triangular:
            raise RuntimeError(("CausalRelativeAttention only supports full "
                                "lower triangular masks"))
        QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "causal-relative", CausalRelativeAttention,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
