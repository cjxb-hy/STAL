import torch
from torch import nn

multi_channel = 1


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0):
        super(ConvBn, self).__init__()

        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, dilation),
            # nn.BatchNorm2d(out_channels)
            nn.InstanceNorm2d(out_channels)
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
        self.conv2 = ConvBn(out_channels, out_channels*2)

        self.extra = nn.Sequential()
        if in_channels != out_channels * 2:
            self.extra = nn.Sequential(
                ConvBn(in_channels, out_channels*2, stride=stride)
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


class Im_Generator(nn.Module):
    def __init__(self, n_channels):
        super(Im_Generator, self).__init__()

        self.stem = nn.Sequential(
            ConvBn(n_channels, 16*multi_channel, kernel_size=3,
                   stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = nn.Sequential(
            Resblk(16*multi_channel, 16*multi_channel, stride=2),
            Resblk(16*multi_channel*2, 16*multi_channel),
        )

        # self.layer2 = nn.Sequential(
        #     Resblk(32*multi_channel, 32*multi_channel,),
        #     Resblk(32*multi_channel*2, 32*multi_channel),
        # )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16*multi_channel*2, 16,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.InstanceNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input):

        input = self.stem(input)
        input = self.layer1(input)
        print(input.shape)
        # input = self.layer2(input)
        input = self.decode(input)
        return input


if __name__ == '__main__':

    net = Im_Generator(1)
    x = torch.randn([2, 1, 512, 512])
    output = net(x)
    print(output.shape)
