import torch



class LeNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_size, dropout):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 2)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear((4 * 4) * (self.hidden_channels * 2), self.hidden_channels, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_channels, self.hidden_channels // 2, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_channels // 2, self.num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
