from architectures import utils
import numpy as np
import torchvision
from torch import nn
import hydra

from architectures.utils import Flatten

mobilenet_cifar_10_setup = dict(
    inverted_residual_setting= [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 1],
    [6, 32, 3, 1],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 1],
    [6, 320, 1, 1]],
    num_channels_per_layer = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
)
mobilenet_imagenet_setup = dict(
    inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
    ], num_channels_per_layer=[32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]
)


def get_mobilenet_parts(pretrained=False, weight_reset=utils.weight_reset):
    cfg = hydra.compose( config_name="config" )
    width = cfg.arch.arch['width']
    num_classes = cfg.dataset.dataset['num_classes']
    part_idx = cfg.quantization.quantization['part_idx']
    decoder_copies = cfg.ensemble.ensemble['n_ensemble']
    mobilenet_setup = cfg.arch.arch['mobilenet_setup']
    if mobilenet_setup == 'CIFAR':
        inverted_residual_setting = mobilenet_cifar_10_setup['inverted_residual_setting']
        num_channels_per_layer = np.array(mobilenet_cifar_10_setup['num_channels_per_layer']) * width
    elif mobilenet_setup == 'IMAGENET':
        inverted_residual_setting = mobilenet_imagenet_setup['inverted_residual_setting']
        num_channels_per_layer = np.array(mobilenet_imagenet_setup['num_channels_per_layer'] ) * width
    m = torchvision.models.mobilenet_v2( pretrained=pretrained, width_mult=width, num_classes=1000,
                                         inverted_residual_setting=inverted_residual_setting )
    torchvision.models.mnasnet0_5()
    out_dict = dict( encoder=[], decoders=[] )
    if part_idx == 1:
        stop_layer = 4
    elif part_idx == 2:
        stop_layer = 7
    elif part_idx == 3:
        stop_layer = 11
    elif part_idx == 4:
        stop_layer = len( m.features ) - 1
    num_channels = int( num_channels_per_layer[stop_layer] )
    classifer = nn.Sequential(
        nn.Dropout( 0.2 ),
        nn.Linear( m.last_channel, num_classes )
    )
    pool = nn.AdaptiveAvgPool2d( (1, 1) )
    encoder_layers = []
    decoder_layers = []
    for layer_idx, l in enumerate( m.features ):
        if layer_idx <= stop_layer:
            encoder_layers.append( l )
        else:
            decoder_layers.append( l )
    decoder_layers.append( pool )
    decoder_layers.append( Flatten() )
    decoder_layers.append( classifer )
    out_dict['encoder'] = nn.Sequential( *encoder_layers )
    out_dict['decoders'] = [nn.Sequential( *decoder_layers )]
    while decoder_copies > 1:
        import copy
        new_model = copy.deepcopy( nn.Sequential( *decoder_layers ) )
        new_model.apply( weight_reset )
        out_dict['decoders'].append( new_model )
        decoder_copies -= 1
    out_dict['num_channels'] = num_channels

    return out_dict