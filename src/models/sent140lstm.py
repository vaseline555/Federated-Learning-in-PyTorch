import torch

from src.models.model_utils import Lambda



class Sent140LSTM(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, hidden_size, dropout, num_layers, glove_emb):
        super(Sent140LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.num_hiddens = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        self.glove_emb = glove_emb
        
        self.features = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(self.glove_emb, padding_idx=0),
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
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_hiddens, self.num_hiddens, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.num_hiddens, self.num_classes, bias=True)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x[:, -1, :])
        return x
