import torch
import torch.nn as nn
from vocabs.vocab import Vocab
from .utils import ViWordEmbedder
from builders.model_builder import META_ARCHITECTURE


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
        # self.softmax = nn.Softmax(dim=-1) 
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
class RNNmodel_ABSA_ViPherV2(nn.Module):
    """
    RNN Model for text classification tasks.
    """
    def __init__(self, config, vocab: Vocab):
        super(RNNmodel_ABSA_ViPherV2, self).__init__()
        # Model configuration
        self.device = config.model.device
        self.input_dim = config.model.input_dim
        self.d_model = config.model.d_model
        self.num_layer = config.model.num_layer
        self.dropout_prob = config.model.dropout
        self.num_output = config.model.num_output
        self.num_categories = config.model.num_categories
        self.bidirectional = config.model.bidirectional
        self.model_type = config.model.model_type
        self.label_smoothing = config.model.label_smoothing

        # Embedding layer
        self.pad_idx = vocab.pad_idx
        self.embedding = ViWordEmbedder(config, vocab)
            
        self.num_labels = config.model.num_output

        # RNN layer
        if self.model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=self.d_model,
                num_layers=self.num_layer,
                bidirectional= True if self.bidirectional==2 else False,
                batch_first= True,
                dropout=self.dropout_prob if self.num_layer > 1 else 0,
            )
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)

        # ABSA output head 
        self.outputHead = Aspect_Based_SA_Output(config.model.dropout  , self.d_model * self.bidirectional , self.num_output, self.num_categories )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing = self.label_smoothing)

    def forward(self, x, labels=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, 3).
            labels (Tensor): Target labels.

        Returns:
            Tuple: (logits, loss)
        """

        # Embedding
        x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)

        # Forward pass
        if 'LSTM' in self.model_type:
            # hn: (num_layers * num_directions, batch_size, hidden_dim)
            _, (hn, _) = self.rnn(x)
        else:
            _, hn = self.rnn(x)
        _, bs, _ = hn.shape
        hn = hn[-2:].permute(1, 2, 0).reshape(bs, -1)

        # Dropout and fully connected layer
        out = self.dropout(hn)
        logits = self.outputHead(out)

        # Compute loss
        if labels is not None:
            loss =self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits