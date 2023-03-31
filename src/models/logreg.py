import torch



class LogReg(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super(LogReg, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, num_classes, bias=True),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x