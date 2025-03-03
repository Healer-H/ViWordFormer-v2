import torch
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
        self.model_type = config.model_type
        self.bidirectional = config.embedder.bidirectional
        self.dropout_prob = config.embedder.dropout
        self.num_layer = config.embedder.num_layer
        self.device = config.device
        self.pad_idx = vocab.get_pad_idx
        self.total_token_dict = vocab.total_tokens_dict

        self.embedding_onset = nn.Embedding(
            num_embeddings=self.total_token_dict["onset"],
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.embedding_tone = nn.Embedding(
            num_embeddings=vocab.total_tokens_dict["tone"],
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )

        self.embedding_nucleus = nn.Embedding(
            num_embeddings=vocab.total_tokens_dict["nucleus"],
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.embedding_medial = nn.Embedding(
            num_embeddings=vocab.total_tokens_dict["medial"],
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )
        self.embedding_coda = nn.Embedding(
            num_embeddings=vocab.total_tokens_dict["coda"],
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_idx,
        )

        if self.model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layer,
                bidirectional=True if self.bidirectional == 2 else False,
                batch_first=True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.embed_dim,
                num_layers=self.num_layer,
                bidirectional=True if self.bidirectional == 2 else False,
                batch_first=True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )

    def forward(self, x):
        """
        (bs, seq_len, 5)
        """
        onset = x[:, :, 0]
        tone = x[:, :, 1]
        medial = x[:, :, 2]
        nucleus = x[:, :, 3]
        coda = x[:, :, 4]

        onset_embed = self.embedding_onset(onset)  # (bs, seq_len, d_model)
        medial_embed = self.embedding_medial(medial)
        nuclues_embed = self.embedding_onset(nucleus)
        coda_embed = self.embedding_onset(coda)
        tone_embed = self.embedding_tone(tone)
        # stack_embed.shape = (bs, seq_len, 5, d_model)
        stack_embed = torch.stack(
            [onset_embed, medial_embed, nuclues_embed, coda_embed, tone_embed], dim=2
        )

        batch_size, seq_len = stack_embed.shape[:2]

        # (bs * seq_len, 5, d_model)
        stack_embed = stack_embed.reshape(
            batch_size * seq_len, stack_embed.shape[2], stack_embed.shape[3]
        )

        h0 = torch.zeros(
            self.num_layer * self.bidirectional,
            batch_size * seq_len,
            self.embed_dim,
            device=self.device,
        )

        if "LSTM" in self.model_type:
            _, (hn, _) = self.rnn(stack_embed, (h0, h0))
        else:
            _, hn = self.rnn(
                stack_embed, h0
            )  # hn: (num_layers * num_directions, batch_size * seq_len, d_model)

        # Extract the last hidden states from both directions
        idx = -self.bidirectional
        hn = hn[idx:]  # Shape: (bidirectional, batch_size * seq_len, d_model)
        hn = hn.permute(1, 0, 2).reshape(
            batch_size, -1
        )  # Shape: (batch_size * seq_len, bidirectional * d_model)

        # (bs, seq_len, d_model)
        hn = hn.reshape(batch_size, seq_len, -1)
        return hn
