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


class AI85Net20(nn.Module):
    """
    CNN that tries to achieve accuracy > %90 for kws.
    """
    def __init__(self, num_classes=32, num_channels=1, dimensions=(64, 64),
                 fc_inputs=64, bias=False, **kwargs):
        super().__init__()

        print(dimensions)

        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 16, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> 15x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 30x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 30x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 60x8x8

        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1,
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


def ai85net20(pretrained=False, **kwargs):
    """
    Constructs a AI84Net20 model.
    """
    assert not pretrained
    return AI85Net20(**kwargs)


models = [
    {
        'name': 'ai85net20',
        'min_input': 1,
        'dim': 2,
    },
]
