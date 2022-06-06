import numpy as np
import torchvision.models as models
from torch import nn
import copy
import torch
from Archs.ResNet18ForCIFAR import resnet18


def weight_noise(m):
    ## Reset all the parameters of the new 'Decoder'.
    ## For creating an ensembles of decoders.
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data = m.weight.data + torch.randn((m.weight).shape)*0.05
        if m.bias is not None:
            m.bias.data = m.bias.data + torch.randn((m.bias.shape))*0.02

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        '''
        TODO: the first dimension is the data batch_size
        so we need to decide how the input shape should be like
        '''
        return input.view((input.shape[0], -1))


### Architecture Settings: (CIFAR10 or ImageNet)

def SplitEffNet(width=1, pretrained=True, num_classes=1000, stop_layer=4,
                   decoder_copies=1, architectura = 'mobilenetv2',pre_train_cifar = True):

    encoder_layers = []
    decoder_layers = []
    EncDec_dict = dict(encoder=[], decoders=[])
    if architectura == 'resnet18':
        model = resnet18(width)
        if stop_layer <= 1:
            residual_stop = 1
        elif stop_layer >= len(resnet18) - 1:
            residual_stop = len(resnet18) - 1
        encoder_layers.append(model.conv1)
        encoder_layers.append(model.conv2_x)
        decoder_layers.append(model.conv3_x)
        decoder_layers.append(model.conv4_x)
        decoder_layers.append(model.conv5_x)
        decoder_layers.append(model.avg_pool)
        decoder_layers.append(Flatten())
        decoder_layers.append(model.fc)

    if architectura == 'mobilenetv2':
        inverted_residual_setting=[[1, 16, 1, 1],[6, 24, 2, 1],[6, 32, 3, 1],[6, 64, 4, 2],[6, 96, 3, 1],[6, 160, 3, 1],[6, 320, 1, 1]]
        num_channels_per_layer = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]

        mobilenetv2 = models.mobilenet_v2(pretrained=pretrained, num_classes=1000,width_mult=width,inverted_residual_setting=inverted_residual_setting)
        res_stop = 5
        for layer_idx, l in enumerate(mobilenetv2.features):
            if layer_idx <= res_stop:
                encoder_layers.append(l)
            else:
                decoder_layers.append(l)

        dropout = nn.Dropout(0.2,inplace=True)
        fc = nn.Linear(in_features=1000,out_features=num_classes,bias=True)
        classifier = nn.Sequential(dropout,fc)
        pool = nn.AdaptiveAvgPool2d(1)
        decoder_layers.append(pool)
        decoder_layers.append(Flatten())
        decoder_layers.append(classifier)








    EncDec_dict['encoder'] = nn.Sequential(*encoder_layers)
    EncDec_dict['decoders'] = [nn.Sequential(*decoder_layers)] # listed for a list of decoders

    ## Creating a list of different Decoders
    while decoder_copies > 1:
        new_decoder = copy.deepcopy(nn.Sequential(*decoder_layers))
        # new_decoder.apply(weight_noise)
        EncDec_dict['decoders'].append(new_decoder)
        decoder_copies -= 1

    if pre_train_cifar == True:
        None

    return EncDec_dict
