import math
import torch
from torch import nn
import torch.nn.functional as F


class D_F(nn.Module):

    def __init__(self, ch_in=4, n_classes=4):
        super(D_F, self).__init__()

        self.ch_in = ch_in
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Conv2d(self.ch_in, 128, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        )

        self.cls1 = nn.Conv2d(512, self.n_classes,
                              kernel_size=1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(512, self.n_classes,
                              kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        _, _, h, w = x.shape

        x = self.model(x)

        out1 = self.cls1(x)
        out2 = self.cls2(x)
        out = torch.cat((out1, out2), dim=1)
        out = F.interpolate(out, size=(h*8, w*8),
                            mode='bilinear', align_corners=False)

        return out


class Dst_Dts(nn.Module):

    def __init__(self, ch_in=1):
        super(Dst_Dts, self).__init__()

        self.ch_in = ch_in
        self.mul = 1

        self.model = nn.Sequential(
            nn.Conv2d(self.ch_in, 16*self.mul,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(16*self.mul, 32*self.mul,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(32*self.mul, 64*self.mul,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64*self.mul, 1, kernel_size=4, stride=2, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.model(x)

        return out


if __name__ == '__main__':

    net = Dst_Dts()
    x = torch.randn([2, 1, 512, 512])
    print(net(x).shape)
