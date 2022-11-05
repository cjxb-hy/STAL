import torch
from torch import nn
multi_channel = 2


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0):
        super(ConvBn, self).__init__()

        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, dilation),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):

        output = self.convbn(input)

        return output


class Resblk(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, padding=None):
        super(Resblk, self).__init__()

        if padding is None:
            padding = 1

        self.conv0 = ConvBn(in_channels, out_channels)
        self.conv1 = ConvBn(out_channels, out_channels,
                            3, stride, dilation, padding)
        self.conv2 = ConvBn(out_channels, out_channels*4)

        self.extra = nn.Sequential()
        if in_channels != out_channels * 4:
            self.extra = nn.Sequential(
                ConvBn(in_channels, out_channels*4, stride=stride)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)

        y = self.extra(input)
        output = torch.add(x, y)

        output = self.relu(output)

        return output


class Resnet50(nn.Module):
    def __init__(self, n_channels):
        super(Resnet50, self).__init__()

        self.stem = nn.Sequential(
            ConvBn(n_channels, 16*multi_channel, kernel_size=7,
                   stride=2, dilation=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            Resblk(16*multi_channel, 16*multi_channel),
            Resblk(16*multi_channel*4, 16*multi_channel),
            # Resblk(16*multi_channel*4, 16*multi_channel)
        )

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(
            Resblk(64*multi_channel, 32*multi_channel, stride=2),
            Resblk(32*multi_channel*4, 32*multi_channel),
            Resblk(32*multi_channel*4, 32*multi_channel),
            Resblk(32*multi_channel*4, 32*multi_channel)
        )

        self.layer3 = nn.Sequential(
            Resblk(128*multi_channel, 64*multi_channel,
                   dilation=2, padding=2),
            Resblk(64*multi_channel*4, 64*multi_channel,
                   dilation=2, padding=2),
            Resblk(64*multi_channel*4, 64*multi_channel, dilation=2, padding=2),
            Resblk(64*multi_channel*4, 64*multi_channel, dilation=2, padding=2),
            Resblk(64*multi_channel*4, 64*multi_channel, dilation=2, padding=2),
            Resblk(64*multi_channel*4, 64*multi_channel, dilation=2, padding=2)
        )

        self.layer4 = nn.Sequential(
            Resblk(256*multi_channel, 128*multi_channel, dilation=1, padding=1),
            Resblk(128*multi_channel*4, 128 *
                   multi_channel, dilation=2, padding=2),
            Resblk(128*multi_channel*4, 128 *
                   multi_channel, dilation=4, padding=4)
        )

        # self.layer5 = nn.Sequential(
        #     Resblk(512*multi_channel, 128*multi_channel, dilation=1, padding=1),
        #     Resblk(128*multi_channel*4, 128 *
        #            multi_channel, dilation=2, padding=2),
        #     Resblk(128*multi_channel*4, 128 *
        #            multi_channel, dilation=4, padding=4)
        # )

        # self.layer6 = nn.Sequential(
        #     Resblk(512*multi_channel, 128*multi_channel,
        #            dilation=2, padding=2),
        #     Resblk(128*multi_channel*4, 128 *
        #            multi_channel, dilation=4, padding=4),
        #     Resblk(128*multi_channel*4, 128 *
        #            multi_channel, dilation=8, padding=8)
        # )

        # self.layer7 = nn.Sequential(
        #     Resblk(512*multi_channel, 128*multi_channel,
        #            dilation=4, padding=4),
        #     Resblk(128*multi_channel*4, 128 *
        #            multi_channel, dilation=8, padding=8),
        #     Resblk(128*multi_channel*4, 128 *
        #            multi_channel, dilation=16, padding=16)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        x = self.stem(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.layer4(x)

        return output
        # input = self.layer5(input)
        # input = self.layer6(input)
        # input = self.layer7(input)


if __name__ == '__main__':

    net = Resnet50(1)
    # print(net)
    x = torch.randn([2, 1, 512, 512])
    x1, x2, x3, output = net(x)
