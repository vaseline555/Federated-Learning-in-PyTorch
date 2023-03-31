import math
import torch

from src.models.model_utils import FireBlock



class SqueezeNet(torch.nn.Module): # MobileNetv3-small
    def __init__(self, in_channels, num_classes, dropout):
        super(SqueezeNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireBlock(64, 16, 64, 64),
            FireBlock(128, 16, 64, 64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireBlock(128, 32, 128, 128),
            FireBlock(256, 32, 128, 128),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireBlock(256, 48, 192, 192),
            FireBlock(384, 48, 192, 192),
            FireBlock(384, 64, 256, 256),
            FireBlock(512, 64, 256, 256),
        )
        final_conv = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            final_conv, 
            torch.nn.ReLU(True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m is final_conv:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x
