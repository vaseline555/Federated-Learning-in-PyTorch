import torch

from src.models.model_utils import MV2Block, MobileViTBlock



class MobileViT(torch.nn.Module):
    L = [2, 4, 3]
    DIMS = [64, 80, 96]
    CHANNELS = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    
    def __init__(self, crop, in_channels, num_classes, dropout):
        super(MobileViT, self).__init__()
        assert (crop is not None) and (crop >= 128) and (crop % 2 == 0)
        
        self.size = crop
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.patch_size = 2
        
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.CHANNELS[0], 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.CHANNELS[0]),
            torch.nn.SiLU(True)
        )
    
        self.mv2 = torch.nn.ModuleList([])
        self.mv2.append(MV2Block(self.CHANNELS[0], self.CHANNELS[1], 1, 4))
        self.mv2.append(MV2Block(self.CHANNELS[1], self.CHANNELS[2], 2, 4))
        self.mv2.append(MV2Block(self.CHANNELS[2], self.CHANNELS[3], 1, 4))
        self.mv2.append(MV2Block(self.CHANNELS[2], self.CHANNELS[3], 1, 4))
        self.mv2.append(MV2Block(self.CHANNELS[3], self.CHANNELS[4], 2, 4))
        self.mv2.append(MV2Block(self.CHANNELS[5], self.CHANNELS[6], 2, 4))
        self.mv2.append(MV2Block(self.CHANNELS[7], self.CHANNELS[8], 2, 4))
        
        self.mvit = torch.nn.ModuleList([])
        self.mvit.append(MobileViTBlock(self.DIMS[0], self.L[0], self.CHANNELS[5], 3, self.patch_size, int(self.DIMS[0] * 2)))
        self.mvit.append(MobileViTBlock(self.DIMS[1], self.L[1], self.CHANNELS[7], 3, self.patch_size, int(self.DIMS[1] * 4)))
        self.mvit.append(MobileViTBlock(self.DIMS[2], self.L[2], self.CHANNELS[9], 3, self.patch_size, int(self.DIMS[2] * 4)))
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.CHANNELS[0], 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.CHANNELS[0]),
            torch.nn.SiLU(True),
            MV2Block(self.CHANNELS[0], self.CHANNELS[1], 1, 4),
            MV2Block(self.CHANNELS[1], self.CHANNELS[2], 2, 4),
            MV2Block(self.CHANNELS[2], self.CHANNELS[3], 1, 4),
            MV2Block(self.CHANNELS[2], self.CHANNELS[3], 1, 4),
            MV2Block(self.CHANNELS[3], self.CHANNELS[4], 2, 4),
            MobileViTBlock(self.DIMS[0], self.L[0], self.CHANNELS[5], 3, 2, int(self.DIMS[0] * 2)),
            MV2Block(self.CHANNELS[5], self.CHANNELS[6], 2, 4),
            MobileViTBlock(self.DIMS[1], self.L[1], self.CHANNELS[7], 3, 2, int(self.DIMS[1] * 4)),
            MV2Block(self.CHANNELS[7], self.CHANNELS[8], 2, 4),
            MobileViTBlock(self.DIMS[2], self.L[2], self.CHANNELS[9], 3, 2, int(self.DIMS[2] * 4)),
            torch.nn.Conv2d(self.CHANNELS[-2], self.CHANNELS[-1], 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(self.CHANNELS[-1]),
            torch.nn.SiLU(True)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AvgPool2d(self.size // 32, 1),
            torch.nn.Flatten(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.CHANNELS[-1], self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
