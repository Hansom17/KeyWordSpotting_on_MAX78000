###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
KWS Networks for AI85

Optionally quantize/clamp activations
"""
from torch import nn

import ai8x


class KWSCNN(nn.Module):
    # KWSCNN has 99785 -> ~100k parameters
    # MAX78000 has 442k 8-bit weight Capacity
    def __init__(self, num_classes=32, num_channels=1, dimensions=(64, 64),
                 fc_inputs=64, networkWidth = 64, bias=False, **kwargs):
        super().__init__()

        print(dimensions)

        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, networkWidth/4, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> 15x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(networkWidth/4, networkWidth/2, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 30x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(networkWidth/2, networkWidth, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 60x8x8

        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(networkWidth, 64, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x4x4

        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(64, 32, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x2x2

        self.conv6 = ai8x.FusedAvgPoolConv2dBNReLU(32, fc_inputs, 3, pool_size=3, pool_stride=3, padding=1, bias=bias, **kwargs)
        
        dim //= 3

        self.fc = ai8x.FusedLinearReLU(fc_inputs*dim*dim, num_classes, bias=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def kwscnn(pretrained=False, **kwargs):
    assert not pretrained
    model = KWSCNN(**kwargs)
    return model

def kwscnn128(pretrained=False, **kwargs):
    assert not pretrained
    return KWSCNN(networkWidth = 128, **kwargs)


models = [
    {
        'name': 'kwscnn',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'kwscnn128',
        'min_input': 1,
        'dim': 2,
    }
]
