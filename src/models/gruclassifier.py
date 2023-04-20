import torch



class GRUClassifier(torch.nn.Module):
    def __init__(self, in_features, hidden_size, num_classes, dropout, num_layers):
        super(GRUClassifier, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.num_layers = num_layers
        self.features = torch.nn.GRU(input_size=self.in_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.classifier = torch.nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)

    def forward(self, x):
        x = self.features(x)[0]
        x = self.classifier(x[:, -1, :])
        return x