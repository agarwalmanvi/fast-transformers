#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import math

import torch
from torch.nn import Dropout, Module, Parameter, functional as F

from ..events import EventDispatcher, AttentionEvent


class SinusoidalPositionalAttention(Module):
    """Exact sinusoidal positional attention."""
    def __init__(self, softmax_temp=None, attention_dropout=0.,
                 event_dispatcher="",
                 num_heads: int = 8,
                 in_features: int = 64,
                 num_sines: int = 1,
                 gated=True):
        super(SinusoidalPositionalAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.num_heads = num_heads
        self.in_features = in_features
        self.num_sines = num_sines
        self.gated = gated

        # register the parameter
        for param in ['freqs', 'offsets', 'gains']:
            self.register_parameter(
                param,
                Parameter(
                    torch.randn(
                        num_heads,
                        in_features,
                        num_sines
                    )
                )
            )

        # normalize the gains
        self.gains.data[...] /= torch.sqrt(
            self.gains.norm(dim=-1, keepdim=True)) / 2.

        # bias initial frequencies to low values for long term range
        self.freqs.data[...] -= 4.

        if gated:
            self.register_parameter('gate', Parameter(
                torch.randn((num_heads, in_features))
            ))

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
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./math.sqrt(E)

        # Copied from SineSPE:

        # build omega_q and omega_k,
        # with shape (length, num_heads, keys_dim, 2*num_sines)
        positions = torch.linspace(0, L-1, L, device=self.freqs.device)

        # making sure the frequencies are in [0, 0.5]
        freqs = torch.sigmoid(self.freqs[None, :, :, :])/2.

        phases_q = (
            2 * math.pi
            * freqs * positions[:, None, None, None]
            + self.offsets[None, :, :, :]
        )
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            L, self.num_heads, self.in_features, 2*self.num_sines
        )

        phases_k = (
            2 * math.pi
            * freqs * positions[:, None, None, None]
        )
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            L, self.num_heads, self.in_features, 2*self.num_sines
        )

        # gains is (num_heads, keys_dim, num_sines). Making then nonnegative with softplus
        gains = F.softplus(self.gains)

        # now upsample it to (num_heads, keys_dim, 2*num_sines)
        gains = torch.stack(
            (gains, gains), dim=-1).view(
                self.num_heads, self.in_features, 2*self.num_sines)

        # compute positional templates
        P = torch.einsum('lhdk,hdk,shdk->lshd', omega_q, gains ** 2, omega_k)

        # apply gating
        if self.gated:
            P = (1 - self.gate) * P + self.gate

        # Copied from FullAttention:

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,lshd,nshd->nhls", queries, P, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()

