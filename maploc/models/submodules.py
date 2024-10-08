import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-12

class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)


class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                                   bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                               bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class FeatExtNet(nn.Module):
    def __init__(self, base_channels, num_stage=3,):
        super(FeatExtNet, self).__init__()

        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, 1, 1, bias=False)
        self.out_channels = [base_channels]
        self.confidence1 = AdaptationBlock(base_channels * 4, 1)

        if num_stage == 3:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
            self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

            self.out2 = nn.Conv2d(base_channels * 2, 1 , 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, 1, 1, bias=False)
            
            self.fine_conv = nn.Sequential(
                nn.Conv2d(base_channels+4, (base_channels+4)*2, 5, padding=2),
                nn.Conv2d((base_channels+4)*2, base_channels+4, 5, padding=2),
                nn.Conv2d(base_channels+4, 1, 1),
                # nn.Sigmoid()
            )
            
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.confidence3 = AdaptationBlock(base_channels    , 1)
            self.out_channels.append(base_channels) # 改
            self.out_channels.append(base_channels)

        elif num_stage == 2:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)

            self.out2 = nn.Conv2d(base_channels * 2, 1, 1, bias=False)
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        conf = self.confidence1(intra_feat).sigmoid()

        outputs["stage1_f"] = out
        outputs["stage1_c"] = conf
        
        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf

            intra_feat = self.deconv2(conv0, intra_feat)
            out = self.out3(intra_feat)
            conf = self.confidence3(intra_feat).sigmoid()
            outputs["stage3_f"] = out
            outputs["stage3_c"] = conf
            inp_fine = torch.cat((intra_feat, out, x), dim=1)
            out_fine = self.fine_conv(inp_fine)
            outputs["stage_fine"] = out_fine

        elif self.num_stage == 2:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf

        return outputs
