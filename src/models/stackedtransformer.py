import math
import torch

from src.models.model_utils import Lambda, PositionalEncoding



class StackedTransformer(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, num_embeddings, hidden_size, seq_len, dropout, num_layers, is_seq2seq):
        super(StackedTransformer, self).__init__()
        self.is_seq2seq = is_seq2seq
        self.num_hiddens = hidden_size
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.features = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_size),
            PositionalEncoding(self.embedding_size, self.dropout),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(self.embedding_size, 16, self.num_hiddens, self.dropout, batch_first=True), 
                self.num_layers
            )
        )
        self.classifier = torch.nn.Linear(self.embedding_size, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x if self.is_seq2seq else x[:, 0, :])
        return x
