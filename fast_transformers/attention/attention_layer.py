#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""The base attention layer performs all the query key value projections and
output projections leaving the implementation of the attention to the inner
attention module.

The transformer layers, however, are agnostic of the attention implementation
and any layer that implements the same interface can substitute for the
attention layer.
"""

import torch
from torch.nn import Linear, Module

from ..events import EventDispatcher, QKVEvent
from ..masking import FullMask, LengthMask


class AttentionLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, positional_encoder=None,
                 event_dispatcher=""):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model, d_keys * n_heads)
        self.value_projection = Linear(d_model, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.positional_encoder = positional_encoder
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, pos_code=None, cache=None, **kwargs):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
            pos_code: The position code to pass to the positional encoder.

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Cache
        Lc, Sc = 0, 0  # The lengths of cached queries and keys
        if cache is not None:
            if self not in cache:
                cache[self] = {}
            if 'outputs' in cache[self]:
                # Remove the cached positions
                Lc = cache[self]['outputs'].shape[1]
                Sc = cache[self]['keys'].shape[1]
                queries = queries[:, Lc:]
                keys, values = keys[:, Sc:], values[:, Sc:]

        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if self.positional_encoder:
            # check if rope is being used
            keys_pos_code = list(pos_code.keys())
            # if RoPE is being used
            if "qhat" not in keys_pos_code:
                q_ = queries.clone().detach().view(N, L, H, -1, 2)
                k_ = keys.clone().detach().view(N, L, H, -1, 2)
                norm_queries = torch.norm(q_, dim=[1, -1]).clone().detach().cpu().numpy()
                norm_keys = torch.norm(k_, dim=[1, -1]).clone().detach().cpu().numpy()
            else:
                norm_queries = torch.norm(queries, dim=[1]).clone().detach().cpu().numpy()
                norm_keys = torch.norm(keys, dim=[1]).clone().detach().cpu().numpy()
        save_objects = {
            "queries": norm_queries,
            "keys": norm_keys
        }
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if cache is not None and cache[self]:
            # Apply positional encoding to new positions
            if self.positional_encoder:
                queries, keys = self.positional_encoder(
                    queries, keys, (pos_code[0][:, Lc:], pos_code[1][:, Sc:]))

            # Restore the cached keys and values
            keys = torch.cat([cache[self]['keys'], keys], dim=1)
            values = torch.cat([cache[self]['values'], values], dim=1)
            S = keys.shape[1]

            # Adjust the masks for queries
            attn_mask = FullMask(attn_mask.bool_matrix[Lc:, :])
            query_lengths = LengthMask(
                query_lengths.lengths - Lc,
                device=query_lengths.lengths.device)
        else:
            # Apply positional encoding
            if self.positional_encoder:
                queries, keys = self.positional_encoder(
                    queries, keys, pos_code)

        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths,
            **kwargs
        ).view(N, L, -1)

        # Project the output
        outputs = self.out_projection(new_values)

        if cache is not None:
            if cache[self]:
                # Add the cached outputs
                outputs = torch.cat([cache[self]['outputs'], outputs], dim=1)

            # Update the cache
            for name in ['keys', 'values', 'outputs']:
                cache[self][name] = locals()[name]

        return outputs, save_objects
