import torch



class LogReg(torch.nn.Module):
    def __init__(self, in_features, num_layers, hidden_size, num_classes):
        super(LogReg, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        if num_layers == 1:
            self.features = torch.nn.Identity()
            self.classifier = torch.nn.Linear(in_features, num_classes, bias=True)
        else:
            features = [torch.nn.Linear(in_features, hidden_size, bias=True)]
            for _ in range(num_layers - 1):
                features.append(torch.nn.Linear(hidden_size, hidden_size, bias=True))
                features.append(torch.nn.ReLU(True))
            self.features = torch.nn.Sequential(*features)
            self.classifier = torch.nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x