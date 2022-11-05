import torch
from torch import nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=256, pd=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=1, stride=1, padding=0)
        )

        self.atrous_block1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.atrous_block6 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=pd[0], dilation=pd[0])
        self.atrous_block12 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=pd[1], dilation=pd[1])
        self.atrous_block18 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=pd[2], dilation=pd[2])
        self.conv_output = nn.Conv2d(
            out_channel * 5, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input):

        b, c, h, w = input.shape

        image_features = self.pool_conv(input)
        image_features = F.interpolate(image_features, size=(h, w),
                                       mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(input)
        atrous_block6 = self.atrous_block6(input)
        atrous_block12 = self.atrous_block12(input)
        atrous_block18 = self.atrous_block18(input)
        output = self.conv_output(torch.cat([image_features, atrous_block1,
                                             atrous_block6, atrous_block12,
                                             atrous_block18], dim=1))

        return output


if __name__ == '__main__':
    net = ASPP(512, 256)
    a = torch.randn([2, 512, 64, 64])
    output = net(a)
    print(output.shape)
