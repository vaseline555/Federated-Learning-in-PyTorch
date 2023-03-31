import torch

from src.models.model_utils import Lambda



class NextWordLSTM(torch.nn.Module):
    def __init__(self, embedding_size, num_embeddings, hidden_size, dropout):
        super(NextWordLSTM, self).__init__()
        self.num_hiddens = hidden_size
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_layers = 2

        self.features = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_size),
            torch.nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.num_hiddens,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bias=True
            ),
            Lambda(lambda x: x[0])
        )
        self.classifier = torch.nn.Linear(self.num_hiddens, self.num_embeddings, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
