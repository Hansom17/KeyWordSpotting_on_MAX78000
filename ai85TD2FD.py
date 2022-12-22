###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Keyword spotting network for AI85/AI86
"""
from torch import nn

import ai8x


class AI85TD2FD(nn.Module):
    """
    Compound KWS20 v3 Audio net, all with Conv1Ds
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=31,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            fc_input = 64,
            **kwargs

    ):
        super().__init__()
        print(num_channels)
        dim = dimensions[0]
        self.drop = nn.Dropout(p=0.2)
        # Time: 256 Feature :64
        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, num_channels, 3, stride=1, padding=1,
                                                bias=bias, **kwargs)
        dim = dim
        # T: 256 F: 64
        self.voice_conv2 = ai8x.FusedConv1dReLU(num_channels, num_channels*2, 3, stride=1, padding=1,
                                                bias=bias, **kwargs)
        
        
        self.voice_convex = ai8x.FusedConv1dReLU(num_channels*2, num_channels*2, 5, stride=1, padding=2,
                                                bias=bias, **kwargs)
        
        
        dim -= 2
        
        # T: 256 F : 128
        self.voice_conv3 = ai8x.FusedAvgPoolConv1d(num_channels*2, num_channels*4, 3, stride=1, padding=1,
                                                       bias=bias, **kwargs)
        dim /= 2
        
        # T: 62 F : 64
        self.voice_conv4 = ai8x.FusedConv1dReLU(num_channels*4, num_channels*6, 3, stride=1, padding=1,
                                                bias=bias, **kwargs)
        
        
        self.seperable = ai8x.FusedConv1dReLU(num_channels*6, num_channels, 1, stride=1,
                                                bias=bias, **kwargs)
        
        
        
        
        
        
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.drop(x)
        x=self.voice_convex(x)
        x = self.voice_conv3(x)
        # print(x.shape)
        x = self.voice_conv4(x)
        
        x = self.seperable(x)
        # print(x.shape)
        

        return x


def td2fd(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    rn AI85KWS20Net(**kwargs)
    """
    assert not pretrained
    return AI85TD2FD(**kwargs)


models = [
    {
        'name': 'td2fd',
        'min_input': 1,
        'dim': 1,
    },
]
