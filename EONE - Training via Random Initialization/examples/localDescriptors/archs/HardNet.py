import torch
from torch import nn
from archs import norms
import kornia

class ExpandLayer( nn.Module):
    def __init__(self, in_channels=128, out_channels=256):
        super( HardNetWithSplit, self ).__init__()
        self.features = nn.Sequential(nn.Linear(in_channels, out_channels, bias=False), nn.BatchNorm1d(out_channels, affine=False), norms.FeatureNorm())
    def forward(self, x):
        return self.features(x)

class HardNetWithSplit( nn.Module ):
    """HardNet model definition
        Assuming input size of 64x64
    """

    def __init__(self, pretrained=True, split=1, expand_layer=nn.Identity):
        super( HardNetWithSplit, self ).__init__()
        self.hardnet_desc = kornia.feature.hardnet.HardNet(pretrained=pretrained)
        # self.exapnd_layer = ExpandLayer(128, 256)
        self.exapnd_layer = expand_layer()
        self.split = split

    def forward(self, input):
        return self.decoder( self.encoder( input ) )

    def encode(self, input):
        features = self.hardnet_desc( input )
        # features = self.exapnd_layer(features)
        return features.view( features.size( 0 ), -1, self.split, 1 )


    def decode(self, input):
        return input.view( input.size( 0 ), -1, self.split )


class HardNet( nn.Module ):
    """HardNet model definition
        Assuming input size of 64x64
    """

    def __init__(self):
        super( HardNet, self ).__init__()
        self.features = nn.Sequential(
            nn.Conv2d( 1, 32, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 32, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 32, 32, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 32, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 32, 64, kernel_size=3, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d( 64, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 64, 64, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 64, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 64, 128, kernel_size=3, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d( 128, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 128, 128, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 128, affine=False ),
            nn.ReLU(),
            nn.Dropout( 0.0 ),
            nn.Conv2d( 128, 128, kernel_size=8, bias=False ),
            nn.BatchNorm2d( 128, affine=False ),
        )
        self.features.apply( weights_init )
        return

    @staticmethod
    def input_norm(x):
        flat = x.view( x.size( 0 ), -1 )
        mp = torch.mean( flat, dim=1 )
        sp = torch.std( flat, dim=1 ) + 1e-7
        return (x - mp.detach().unsqueeze( -1 ).unsqueeze( -1 ).unsqueeze( -1 ).expand_as( x )) / sp.detach().unsqueeze(
            -1 ).unsqueeze( -1 ).unsqueeze( 1 ).expand_as( x )

    def forward(self, input):
        return self.decoder( self.encoder( input ) )

    def encode(self, input):
        return self.features( self.input_norm( input ) )

    def decode(self, input):
        x = input.view( input.size( 0 ), -1 )
        return norms.L2Norm()( x )

class SplittingHead( nn.Module ):
    def __init__(self, input_channels=128, split: int = -1):
        super(SplittingHead, self ).__init__()
        self.split = split
        self.input_channels = input_channels
        self.features = nn.Sequential(FixedPoolingHead(input_channels=self.input_channels),
                                      nn.AdaptiveAvgPool2d( (1, 1) ),
                                        nn.BatchNorm2d( self.input_channels, affine=False ),
                                        nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        if self.split <= 0:
            return self.features(x)
        else:
            return x.view(x.shape[0], self.input_channels//self.split, -1, 1)

class FixedPoolingHead( nn.Module ):
    def __init__(self, pooling_size=(5, 5), input_channels=128):
        super( FixedPoolingHead, self ).__init__()
        self.pooling_size = pooling_size
        self.input_channels = input_channels
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d( self.pooling_size ),
                                        nn.BatchNorm2d(self.input_channels, affine=False ),
                                        nn.ReLU(),)
                                        # norms.FeatureNorm())

    def forward(self, x):
        return self.features(x)



class GlobalPoolingDecoder( nn.Module ):
    def __init__(self, input_channels=128):
        super( GlobalPoolingDecoder, self ).__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d( input_channels, input_channels, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( input_channels, affine=False ),
            nn.AdaptiveAvgPool2d( (1, 1) ),
            nn.BatchNorm2d( 128, affine=False ),
        )
    def forward(self, x):
        return self.features( x )


class HardNetParted( nn.Module ):
    def __init__(self):
        super( HardNetParted, self ).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d( 1, 32, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 32, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 32, 32, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 32, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 32, 64, kernel_size=3, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d( 64, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 64, 64, kernel_size=3, padding=1, bias=False ),
            nn.BatchNorm2d( 64, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 64, 128, kernel_size=3, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d( 128, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 128, 128, kernel_size=3, stride=1, padding=1, bias=False ),
            nn.BatchNorm2d( 128, affine=False ),
            nn.ReLU(),
            nn.Conv2d( 128, 128, kernel_size=3, padding=1, bias=False ),
            # head:
            SplittingHead(split = -1),
            # nn.AdaptiveAvgPool2d( (4, 4) ),
            # nn.BatchNorm2d( 128, affine=False ),
            # nn.ReLU(),
            # norms.FeatureNorm(),  # TODO: normalize per feature

        )
        self.decoder = GlobalPoolingDecoder()
        # self.decoder = nn.Sequential(
        #     # nn.ReLU(),
        #     # nn.Conv2d( 128, 128, kernel_size=3, padding=1, bias=False ),
        #     # nn.BatchNorm2d( 128, affine=False ),
        #     nn.AdaptiveAvgPool2d( (1, 1) ),
        #     # nn.Conv2d( 128, 128, kernel_size=5, padding=0, bias=False ),
        #     # nn.BatchNorm2d( 128, affine=False ),
        # )
        self.encoder.apply( weights_init )
        self.decoder.apply( weights_init )

    def forward(self, x):
        return self.decoder( self.encoder( x ) )

    def encode(self, x):
        return self.encoder( HardNet.input_norm( x ) )

    def decode(self, x):
        x = self.decoder( x )
        x = x.view( x.size( 0 ), -1 )
        return norms.L2Norm()( x )


def weights_init(m):
    if isinstance( m, nn.Conv2d ):
        nn.init.orthogonal_( m.weight.data, gain=0.6 )
        try:
            nn.init.constant( m.bias.data, 0.01 )
        except:
            pass
    return
