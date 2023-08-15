import torch

from src.models.model_utils import ShuffleNetInvRes



class ShuffleNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, dropout):
        super(ShuffleNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        
        # building feature extractor
        features = [] 

        # input layers
        hidden_channels = self.stage_out_channels[1]
        features.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, hidden_channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(hidden_channels),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        )
        
        # inverted residual layers
        for idx, num_repeats in enumerate(self.stage_repeats):
            out_channels = self.stage_out_channels[idx + 2]
            for i in range(num_repeats):
                if i == 0:
                    features.append(ShuffleNetInvRes(hidden_channels, out_channels, 2, 2))
                else:
                    features.append(ShuffleNetInvRes(hidden_channels, out_channels, 1, 1))
                hidden_channels = out_channels

        # pooling layers
        features.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(hidden_channels, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(self.stage_out_channels[-1]),
                torch.nn.ReLU(True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
        )
        self.features = torch.nn.Sequential(*features)              
        self.classifier = torch.nn.Linear(self.stage_out_channels[-1], self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x