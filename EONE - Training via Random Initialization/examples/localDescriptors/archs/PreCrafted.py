import torch
from torch import nn
import kornia

class SiftNet( nn.Module ):
    """SiftNet model definition
        Assuming input size of 64x64
    """

    def __init__(self, num_ang_bins=8, num_spatial_bins=4):
        super( SiftNet, self ).__init__()
        self.sift_desc = kornia.SIFTDescriptor(32, num_ang_bins, num_spatial_bins)

    def forward(self, input):
        return self.decoder( self.encoder( input ) )

    def encode(self, input):
        return self.sift_desc( input ).view( input.size( 0 ), -1, 1, 1 )

    def decode(self, input):
        return input.view( input.size( 0 ), -1 )

class MKDDNet( nn.Module ):
    """MKDDNet model definition
        Assuming input size of 64x64
    """

    def __init__(self, output_dim=128, training_set='liberty'):
        super( MKDDNet, self ).__init__()
        self.mkdd_desc = kornia.MKDDescriptor(32, output_dims=output_dim, training_set=training_set)

    def forward(self, input):
        return self.decoder( self.encoder( input ) )

    def encode(self, input):
        return self.mkdd_desc( input ).view( input.size( 0 ), -1, 1, 1 )

    def decode(self, input):
        return input.view( input.size( 0 ), -1 )


class HardNet( nn.Module ):
    """HardNet model definition
        Assuming input size of 64x64
    """

    def __init__(self, pretrained=True):
        super( HardNet, self ).__init__()
        self.hardnet_desc = kornia.feature.hardnet.HardNet(pretrained=pretrained)

    def forward(self, input):
        return self.decoder( self.encoder( input ) )

    def encode(self, input):
        return self.hardnet_desc( input ).view( input.size( 0 ), -1, 1, 1 )

    def decode(self, input):
        return input.view( input.size( 0 ), -1 )

class HardNet8( nn.Module ):
    """HardNet8 model definition
        Assuming input size of 64x64
    """

    def __init__(self, pretrained=True):
        super( HardNet8, self ).__init__()
        self.hardnet_desc = kornia.feature.hardnet.HardNet8(pretrained=pretrained)

    def forward(self, input):
        return self.decoder( self.encoder( input ) )

    def encode(self, input):
        return self.hardnet_desc( input ).view( input.size( 0 ), -1, 1, 1 )

    def decode(self, input):
        return input.view( input.size( 0 ), -1 )