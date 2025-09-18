import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE
from .utils import ViWordEmbedder

class Aspect_Based_SA_Output(nn.Module): 
    def __init__(self, dropout , d_input, d_output, num_categories):
        """
        Initialization 
        dropout: dropout percent
        d_input: Model dimension 
        d_output: output dimension 
        categories: categories list
        """
        super(Aspect_Based_SA_Output, self).__init__()
        self.dense = nn.Linear(d_input , d_output *num_categories ,  bias=True)
        self.norm = nn.LayerNorm(d_output, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.num_categories = num_categories
        self.num_labels= d_output

    def forward(self, model_output ):
        """ 
         x : Model output 
      
         Output: sentiment output 
        """
       
        x = self.dropout(model_output)
        output = self.dense(x) 
        output = output.view(-1 ,self.num_categories, self.num_labels )
        
        return output
    


@META_ARCHITECTURE.register()
class TextCNN_ABSA_ViPherV2(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super(TextCNN_ABSA_ViPherV2, self).__init__()
        # Model configuration
        self.device = config.model.device
        self.vocab_size = vocab.total_tokens
        self.d_model = config.model.embedding_dim
        self.n_filters = config.model.n_filters
        self.filter_sizes = config.model.filter_sizes
        self.output_dim = config.model.num_output
        self.dropout = config.model.dropout
        self.label_smoothing = config.model.label_smoothing
        self.num_categories = config.model.num_categories
        self.pad_idx = vocab.pad_idx

        # Embedding layer
        self.embedding = ViWordEmbedder(config, vocab)
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.d_model,
                      out_channels=self.d_model,
                      kernel_size=filter_size)
            for filter_size in self.filter_sizes
        ])
        
        # Others layer
        self.dropout = nn.Dropout(self.dropout)
        
        self.cls_head = Aspect_Based_SA_Output(
            config.model.dropout, 
            self.d_model*3, 
            self.output_dim, 
            self.num_categories
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing)

    def forward(self, x, labels=None):
        # x shape: (batch size, sentence length)

        embedded = self.embedding(x) # (batch_size, seq_len, d_model)
        embedded = embedded.permute(0, -1, 1) # (batch_size, d_model, seq_len)

        # Convolutions and average-pooling-over-time
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # [(N, C, L),..] -> [(N, C, 1),..] -> [(N, C),..]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(-1) for conv in conved]

        # Concatenate pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.cls_head(cat)

        if labels is not None:
            with autocast("cuda"):
                loss = self.loss_fn(logits.view(-1, self.output_dim), labels.view(-1))
            return logits, loss

        return logits
