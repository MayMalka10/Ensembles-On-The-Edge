import torch
from torch import nn
from archs import norms
import kornia
import torchvision
import torch.nn.functional as F
from collections import OrderedDict


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None, padding=None, do_relu=True):
        if padding is None:
            padding = (kernel_size - 1) // 2
        else:
            padding = padding
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers = [
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)),
                ('bn', norm_layer(out_planes)),
            ]
        if do_relu:
            layers.append(('relu', nn.ReLU(inplace=True)))
        super(ConvBNReLU, self).__init__(
            OrderedDict(layers)
        )

class IRFBlock( nn.Module ):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super( IRFBlock, self ).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.pw = ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
        self.dw = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer)
        self.pwl = ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, padding=0, do_relu=False)
        # layers = []
        # if expand_ratio != 1:
        #     # pw
        #     layers.append(('pw', ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)))
        # layers.extend([
        #     # dw
        #     ('dw', ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer)),
        #     # pw-linear
        #     ('pwl', ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, padding=0, do_relu=False))
        # ])
        # self.conv = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        y = self.pwl(self.dw(self.pw(x)))
        if self.use_res_connect:
            return x + y
        else:
            return y

class FBNet( nn.Module ):
    """FBNet model definition
        Assuming input size of 32x32
    """

    def __init__(self, split):
        super( FBNet, self ).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(1, 8, 3),
            IRFBlock( 8, 8, stride=2, expand_ratio=3 ),
            IRFBlock( 8, 8, stride=1, expand_ratio=3 ),
            IRFBlock( 8, 8, stride=1, expand_ratio=3 ),
            ConvBNReLU(8, 32, 3, stride=2)
        )
        self.linear = nn.Linear(2048, 64, bias=False)
        self.split = split
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
        return self.decode( self.encode( input ) )

    def encode(self, input):
        x = self.features( self.input_norm( input ) )
        x = x.view(input.size(0), 2048)
        x = self.linear(x)
        x = F.normalize( x, dim=1 )
        return x.view( input.size( 0 ), -1, self.split, 1 )

    def decode(self, input):
        x = input.view( input.size( 0 ), -1)
        return x


def weights_init(m):
    if isinstance( m, nn.Conv2d ):
        nn.init.orthogonal_( m.weight.data, gain=0.6 )
        try:
            nn.init.constant( m.bias.data, 0.01 )
        except:
            pass
    return


class FBNetV2Backbone(nn.Module):
    def __init__(self):
        super( FBNetV2Backbone, self ).__init__()
        self.trunk0 = nn.Sequential(OrderedDict([('fbnetv2_0_0',ConvBNReLU(1, 8, 3))]))
        self.trunk1 = nn.Sequential(OrderedDict([('fbnetv2_1_0', IRFBlock( 8, 8, stride=2, expand_ratio=3 )),
                                    ('fbnetv2_1_1', IRFBlock( 8, 8, stride=1, expand_ratio=3 )),
                                    ( 'fbnetv2_1_2', IRFBlock( 8, 8, stride=1, expand_ratio=3 ))]))
        self.trunk2 = nn.Sequential(OrderedDict([('fbnetv2_2_0', ConvBNReLU(8, 32, 3, stride=2))]))


    def forward(self, x):
        x = self.trunk0(x)
        x = self.trunk1( x )
        return self.trunk2( x )


class DescriptorTrunk(nn.Module):
    def __init__(self, split):
        super( DescriptorTrunk, self ).__init__()
        self.fbnet_trunk = FBNetV2Backbone()
        self.fc_layer = nn.Linear( 2048, 128, bias=False )
        self.split = split
        self.fbnet_trunk.apply( weights_init )
        self.fc_layer.apply( weights_init )

    def forward(self, input):
        return self.decode( self.encode( input ) )

    def encode(self, input):
        x = self.fbnet_trunk(input)
        x = x.view(input.size(0), 2048)
        x = self.fc_layer(x)
        x = F.normalize( x, dim=1 )
        return x.view( input.size( 0 ), -1, self.split, 1 )

    def decode(self, input):
        x = input.view( input.size( 0 ), -1)
        return x

class FBNetPatchBackbone(nn.Module):

    def __init__(self, split=1):
        super( FBNetPatchBackbone, self ).__init__()
        self.input_norm = torch.nn.InstanceNorm2d(1, eps=1e-07, momentum=0.1, affine=False, track_running_stats=False)
        self.body = DescriptorTrunk(split=split)
        self.split = split
        self.body.apply( weights_init )
        return

    def forward(self, input):
        return self.decode( self.encode( input ) )

    def encode(self, input):
        return self.body.encode(self.input_norm(input))

    def decode(self, input):
        return self.body.decode( input )


def strip_prefix(state_dict, prefix='backbone'):
    new_state_dict = {}
    for k in state_dict:
        if k.startswith(prefix):
            new_state_dict[k.partition('.')[2]] = state_dict[k]
    return new_state_dict

# if __name__ == '__main__':
#     net = FBNetPatchBackbone()
#     state_dict = torch.load('/home/erez/Downloads/model_final.pth')
#     net.load_state_dict( strip_prefix( state_dict['model'], 'backbone' ), strict=False )
#     net = net