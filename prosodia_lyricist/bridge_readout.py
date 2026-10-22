"""Parallel position queries over contextual melody memory, without label feedback."""

from torch import nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, memory, memory_key_padding_mask):
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(attended)
        return queries + self.dropout(self.feedforward(self.feedforward_norm(queries)))


class ParallelReadout(nn.Module):
    """Each requested position independently retrieves from the complete song encoding.

    Layers refine queries through cross-attention and feedforward blocks. There is
    no query self-attention, causal mask, token shifting, or prediction feedback.
    """

    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            CrossAttentionLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, queries, memory, *, tgt_key_padding_mask, memory_key_padding_mask):
        for layer in self.layers:
            queries = layer(queries, memory, memory_key_padding_mask)
        return self.norm(queries).masked_fill(tgt_key_padding_mask[..., None], 0)
