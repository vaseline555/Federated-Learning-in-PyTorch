import torch



class FEMNISTCNN(torch.nn.Module): # for FEMNIST experiment in Caldas et al., 2018; (https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
    def __init__(self, in_channels, hidden_size, num_classes):
        super(FEMNISTCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=5, padding=1, stride=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=5, padding=1, stride=1, bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=self.hidden_channels * 2 * (7 * 7), out_features=2048, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
