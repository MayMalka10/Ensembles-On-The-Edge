from torch import nn
import copy
import torch

def add_noise_to_weights(m, sigma=0.01):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * sigma)

def get_noised_model(m):
    m_copy = copy.deepcopy( m )
    m_copy.apply(add_noise_to_weights)
    return m_copy

def add_noise_to_model(m):
    m.apply( add_noise_to_weights )



def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        '''
        TODO: the first dimension is the data batch_size
        so we need to decide how the input shape should be like
        '''
        return input.view((input.shape[0], -1))