import torch
from torch.nn import functional as F
import math
from torch import nn
from vocabs.vocab import Vocab


def generate_padding_mask(seq, pad_token_id=0):
    if seq.ndim == 3:
        """(bs, seq_len, 3) -> padding = [[[pad_token_id, pad_token_id, pad_token_id]]]"""
        pad_token = (
            torch.Tensor([pad_token_id, pad_token_id, pad_token_id])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(seq.device)
        )
        return (seq == pad_token).all(dim=-1)
    return seq == pad_token_id


def new_generate_padding_mask(seq, pad_token_id=0):
    # TODO: update `generate_padding_mask` instead of create new ones.
    if seq.ndim == 3 and seq.shape[-1] == 5:
        """(batch_size, seq_len, 5) -> padding mask"""
        pad_token = torch.full((1, 1, 5), pad_token_id, device=seq.device)
        return (seq == pad_token).all(dim=-1)

    return seq == pad_token_id


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ViWordEmbedder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.embed_dim = config.embedder.embed_dim
        self.model_type = config.embedder.model_type
        self.dropout_prob = config.embedder.dropout
        self.num_layer = config.embedder.num_layer
        self.device = config.model.device
        self.pad_idx = vocab.pad_idx
        self.total_tokens = vocab.total_tokens

        self.embedding = nn.Embedding(
            num_embeddings=self.total_tokens,
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim
        )

        if self.model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layer,
                bidirectional=False,
                batch_first=True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layer,
                bidirectional=False,
                batch_first=True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )

    def forward(self, x):
        """
        (bs, seq_len, 5)
        """

        embedded = self.embedding(x)  # (bs, seq_len, 5, d_model)
        embedded = F.gelu(embedded)

        # turn the tensor into (bs*seq_len, 5, d_model)
        bs, seq_len, dim_1, dim_2 = embedded.shape
        embedded = embedded.reshape((-1, dim_1, dim_2))

        if self.model_type == "LSTM":
            _, (embedded, _) = self.rnn(embedded)
        else:
            _, embedded = self.rnn(embedded)
        
        embedded = embedded[-1]

        # turn the tensor back to (bs, seq_len, d_model)
        embedded = embedded.reshape((bs, seq_len, -1))

        return embedded
