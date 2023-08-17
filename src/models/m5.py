import torch



class M5(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(M5, self).__init__()
        self.in_channels = in_channels
        self.num_hiddens = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(self.in_channels, self.num_hiddens, kernel_size=80, stride=16),
            torch.nn.BatchNorm1d(self.num_hiddens),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(self.num_hiddens, self.num_hiddens, kernel_size=3),
            torch.nn.BatchNorm1d(self.num_hiddens),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(self.num_hiddens, self.num_hiddens * 2, kernel_size=3),
            torch.nn.BatchNorm1d(self.num_hiddens * 2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(self.num_hiddens * 2, self.num_hiddens * 2, kernel_size=3),
            torch.nn.BatchNorm1d(self.num_hiddens * 2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool1d(4),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Linear(self.num_hiddens * 2, self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x