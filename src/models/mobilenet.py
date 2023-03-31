import torch

from src.models.model_utils import make_divisible, SELayer, InvertedResidualBlock



class MobileNet(torch.nn.Module): # MobileNetv3-small
    CONFIG = [# k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1]
    ]
    
    def __init__(self, in_channels, num_classes, dropout):
        super(MobileNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        hidden_channels = make_divisible(16, 8)
        layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, hidden_channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(make_divisible(16, 8)),
                torch.nn.Hardswish(True)
            )
        ]

        # building inverted residual blocks
        for k, t, c, use_se, use_hs, s in self.CONFIG:
            out_channels = make_divisible(c * 1, 8)
            exp_size = make_divisible(hidden_channels * t, 8)
            layers.append(InvertedResidualBlock(hidden_channels, exp_size, out_channels, k, s, use_se, use_hs))
            hidden_channels = out_channels
        else:
            self.features1 = torch.nn.Sequential(*layers)
        
        # building last several layers
        self.features2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, exp_size, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(exp_size),
            torch.nn.Hardswish(True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        out_channels = 1024
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(exp_size, out_channels),
            torch.nn.Hardswish(True),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(out_channels, self.num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.classifier(x)
        return x
