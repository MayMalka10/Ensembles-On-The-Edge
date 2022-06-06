from architectures import utils
from torch import nn
from pl_bolts.models.vision import unet
from segmentation_models_pytorch import pspnet
import copy


class EncoderLast( nn.Module ):
    def __init__(self, encoder):
        super( EncoderLast, self ).__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder( x )[-1]


def get_psp_parts(num_classes, decoder_copies=1, weight_reset=utils.weight_reset, encoder_name="resnet34",
                  pretrained=True):
    encoder_weights = "imagenet" if pretrained else None
    out_dict = dict( encoder=[], decoders=[] )
    net = pspnet.PSPNet( classes=num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights )
    out_dict['encoder'] = EncoderLast( net.encoder )
    out_dict['decoders'] = [nn.Sequential( *[net.decoder, net.segmentation_head] )]
    while decoder_copies > 1:
        new_model = copy.deepcopy( nn.Sequential( *[net.decoder, net.segmentation_head] ) )
        new_model.apply( weight_reset )
        out_dict['decoders'].append( new_model )
        decoder_copies -= 1
    return out_dict


def get_unet_parts(num_classes, decoder_copies=1, weight_reset=utils.weight_reset):
    out_dict = dict( encoder=[], decoders=[] )
    net = unet.UNet( num_classes=num_classes, bilinear=True )
    encoder_layers = net.layers[1:net.num_layers]
    decoder_layers = net.layers[net.num_layers:-1]
    out_dict['encoder'] = nn.Sequential( *encoder_layers )
    out_dict['decoders'] = [nn.Sequential( *decoder_layers )]
    while decoder_copies > 1:
        new_model = copy.deepcopy( nn.Sequential( *decoder_layers ) )
        new_model.apply( weight_reset )
        out_dict['decoders'].append( new_model )
        decoder_copies -= 1
    return out_dict
