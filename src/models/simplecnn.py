import torch



class SimpleCNN(torch.nn.Module): # for CIFAR10 experiment in McMahan et al., 2016; (https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10.py)
    def __init__(self, in_channels, hidden_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2, stride=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.LocalResponseNorm(size=9, alpha=0.001),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=5, padding=2, stride=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.LocalResponseNorm(size=9, alpha=0.001),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((6, 6)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=self.hidden_channels * (6 * 6), out_features=384, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=384, out_features=192, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=192, out_features=self.num_classes, bias=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
