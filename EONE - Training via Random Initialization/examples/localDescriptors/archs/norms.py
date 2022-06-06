import torch
from torch import nn

class FeatureNorm(nn.Module):
    def __init__(self):
        super(FeatureNorm,self).__init__()
        self.eps = 1e-4
    def forward(self, x):
        xn = torch.norm( x, p=2, dim=1).detach().unsqueeze(dim=1)
        x = x.div(xn + self.eps)
        return x

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x