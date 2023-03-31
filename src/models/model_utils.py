import math
import torch
import einops



##########################
# MobileNet & MobileNeXt #
##########################
def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(torch.nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, make_divisible(in_channels // reduction, 8)),
            torch.nn.ReLU(True),
            torch.nn.Linear(make_divisible(in_channels // reduction, 8), in_channels),
            torch.nn.Hardsigmoid(True)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, inputs, hidden_dims, ouputs, kernel_size, stride, use_se, use_hardswish):
        super(InvertedResidualBlock, self).__init__()
        self.identity = stride == 1 and inputs == ouputs

        if inputs == hidden_dims:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dims, bias=False), # depth-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                SELayer(hidden_dims) if use_se else torch.nn.Identity(), # squeeze-excite block
                torch.nn.Conv2d(hidden_dims, ouputs, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(ouputs),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inputs, hidden_dims, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                torch.nn.Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dims, bias=False), # depth-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                SELayer(hidden_dims) if use_se else torch.nn.Identity(), # squeeze-excite block
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                torch.nn.Conv2d(hidden_dims, ouputs, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(ouputs),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SandGlassLayer(torch.nn.Module):
    def __init__(self, inputs, outputs, stride, reduction_ratio):
        super(SandGlassLayer, self).__init__()
        hidden_dim = round(inputs // reduction_ratio)
        self.identity = stride == 1 and inputs == outputs

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(inputs, inputs, 3, 1, 1, groups=inputs, bias=False), # depth-wise convolution
            torch.nn.BatchNorm2d(inputs),
            torch.nn.ReLU6(True),
            torch.nn.Conv2d(inputs, hidden_dim, 1, 1, 0, bias=False), # point-wise convolution
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.Conv2d(hidden_dim, outputs, 1, 1, 0, bias=False), # point-wise convolution
            torch.nn.BatchNorm2d(outputs),
            torch.nn.ReLU6(True),
            torch.nn.Conv2d(outputs, outputs, 3, stride, 1, groups=outputs, bias=False), # depth-wise convolution
            torch.nn.BatchNorm2d(outputs),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
##############
# SqueezeNet #
##############
class FireBlock(torch.nn.Module):
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireBlock, self).__init__()
        self.squeeze_activation = torch.nn.ReLU(True)
        self.in_planes = in_planes
        self.squeeze = torch.nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.expand1x1 = torch.nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = torch.nn.ReLU(True)
        self.expand3x3 = torch.nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)

###############
# SqueezeNeXt #
###############
class SNXBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction=0.5):
        super(SNXBlock, self).__init__()
        if stride == 2:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
        
        self.act = torch.nn.ReLU(True)
        self.squeeze = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction * 0.5)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

        if stride == 2 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Identity()
            
    def forward(self, x):
        out = self.squeeze(x)
        out += self.act(self.shortcut(x))
        out = self.act(out)
        return out

#############
# MobileViT #
#############
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.SiLU(True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.ff(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**(-0.5)

        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(torch.nn.Module):
    def __init__(self, inputs, ouputs, stride=1, expansion=4):
        super(MV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inputs * expansion)
        self.use_res_connect = self.stride == 1 and inputs == ouputs

        if expansion == 1:
            self.conv = torch.nn.Sequential(
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, ouputs, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ouputs),
            )
        else:
            self.conv = torch.nn.Sequential(
                # pw
                torch.nn.Conv2d(inputs, hidden_dim, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(True),
                # dw
                torch.nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.SiLU(True),
                # pw-linear
                torch.nn.Conv2d(hidden_dim, ouputs, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ouputs),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(torch.nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.patch_size = patch_size

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel, kernel_size, 1, 1, bias=False),
            torch.nn.BatchNorm2d(channel),
            torch.nn.SiLU(True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(channel, dim, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(dim),
            torch.nn.SiLU(True)
        )
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(dim, channel, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(channel),
            torch.nn.SiLU(True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(2 * channel, channel, kernel_size, 1, 1, bias=False),
            torch.nn.BatchNorm2d(channel),
            torch.nn.SiLU(True)
        )
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = einops.rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.patch_size, pw=self.patch_size)
        x = self.transformer(x)
        x = einops.rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.patch_size, w=w // self.patch_size, ph=self.patch_size, pw=self.patch_size)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

##########
# ResNet #
##########
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(planes),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(planes)
        )

        self.shortcut = torch.nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        x = self.features(x) + self.shortcut(x) 
        x = torch.nn.functional.relu(x)
        return x

################
# Lambda Layer #
################
class Lambda(torch.nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x): 
        return self.func(x)
