import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from .utils import ViWordEmbedder


@META_ARCHITECTURE.register()
class TextCNN_VipherV2(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TextCNN_VipherV2, self).__init__()
        # Model configuration
        self.device = config.model.device
        self.vocab_size = vocab.total_tokens
        self.d_model = config.model.embedding_dim
        self.n_filters = config.model.n_filters
        self.filter_sizes = config.model.filter_sizes
        self.output_dim = config.model.num_output
        self.dropout = config.model.dropout
        self.label_smoothing = config.model.label_smoothing
        self.pad_idx = vocab.get_pad_idx
        self.total_token_dict = vocab.total_tokens_dict

        self.embedding = ViWordEmbedder(config, vocab)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=self.n_filters,
                      kernel_size=(fs, self.d_model))
            for fs in self.filter_sizes
        ])
        
        # Others layer
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(len(self.filter_sizes) *
                            self.n_filters, self.output_dim)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        # x shape: (batch size, sentence length)

        embedded = self.embedding(x)  # (batch_size, seq_len, d_model)

        # convert to shape: (batch size, 1, sentence length, embedding dim)
        embedded = embedded.unsqueeze(1)

        # Convolutions and max-pooling-over-time
        # after conv: (bs, n_filter, seq_len - filter_size + 1, 1) -squeeze(3)
        # -> (bs, n_filter, seq_len - filter_size + 1) ~ (N, C, L)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # [(N, C, L),..] -> [(N, C, 1),..] -> [(N, C),..]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        # Concatenate pooled features
        # (N, n_filters * len(filter_sizes))
        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.fc(cat)

        if labels is not None:
            loss = self.loss_fn(logits, labels.squeeze(-1))
            return logits, loss

        return logits