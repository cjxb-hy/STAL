import torch
from torch import nn
import torch.nn.functional as F

from .resnet50 import Resnet50
from .aspp import ASPP


class Deeplabv3(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Deeplabv3, self).__init__()

        self.feature_extractor = Resnet50(n_channels)

        self.aspp = ASPP(1024, 256, [6, 12, 18])

        self.output_seg = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n_classes, kernel_size=1, stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        _, _, h, w = input.shape

        feature = self.feature_extractor(input)
        aspp = self.aspp(feature)
        seg_output = self.output_seg(aspp)
        seg_output = F.interpolate(seg_output, size=(h, w),
                                   mode='bilinear', align_corners=False)

        return feature, seg_output


if __name__ == '__main__':
    net = Deeplabv3(1, 4)
    x = torch.randn([2, 1, 512, 512])
    f, output = net(x)
    print(f.shape)
    print(output.shape)
