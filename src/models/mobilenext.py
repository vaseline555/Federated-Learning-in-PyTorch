import math
import torch

from src.models.model_utils import make_divisible, SandGlassLayer



class MobileNeXt(torch.nn.Module):
    CONFIG = [# t, c, n, s
        [2,   96, 1, 2],
        [6,  144, 1, 1],
        [6,  192, 3, 2],
        [6,  288, 3, 2],
        [6,  384, 4, 1],
        [6,  576, 4, 2],
        [6,  960, 3, 1],
        [6, 1280, 1, 1]
    ]
    
    def __init__(self, in_channels, num_classes, dropout):
        super(MobileNeXt, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

        # building first layer
        hidden_channels = make_divisible(32, 8)
        layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, hidden_channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(hidden_channels),
                torch.nn.ReLU6(True)
            )
        ]
        
        # building blocks
        for t, c, n, s in self.CONFIG:
            out_channels = make_divisible(c, 8)
            for i in range(n):
                layers.append(SandGlassLayer(hidden_channels, out_channels, s if i == 0 else 1, t))
                hidden_channels = out_channels
        self.features = torch.nn.Sequential(*layers)
        
        # building classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(out_channels, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
