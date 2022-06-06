import torch
from torch import nn
import sys

def distance_matrix_word_wise(anchor2, positive2):

    dist_matrix = torch.zeros(anchor2.shape[0], anchor2.shape[0], device=anchor2.device)
    for k in range(positive2.shape[2]):
        dist_matrix += distance_matrix_vector(anchor2[:,:,k], positive2[:,:,k], sqrt=False)
    return torch.sqrt(dist_matrix)

def distance_matrix_vector(anchor, positive, sqrt=True):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)
    eps = 1e-4
    m = (d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps
    if sqrt:
        return torch.sqrt(m)
    else:
        return m

class LossHardNet(nn.Module):
    def __init__(self, anchor_swap=False, margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin", split=False):
        super().__init__()
        self.anchor_swap = anchor_swap
        self.margin = margin
        self.batch_reduce = batch_reduce
        self.loss_type = loss_type
        self.eps = 1e-8
        self.split = split


    def forward(self, anchor, positive):
        assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
        if self.split:
            assert anchor.dim() == 3, "Inputd must be a 3D matrix."
            dist_matrix = distance_matrix_word_wise( anchor, positive) + self.eps
        else:
            assert anchor.dim() == 2, "Inputd must be a 2D matrix."
            dist_matrix = distance_matrix_vector( anchor, positive ) + self.eps
        eye = torch.autograd.Variable( torch.eye( dist_matrix.size( 1 ) ) ).cuda()

        # steps to filter out same patches that occur in distance matrix as negatives
        pos1 = torch.diag( dist_matrix )
        dist_without_min_on_diag = dist_matrix + eye * 10
        mask = (dist_without_min_on_diag.ge( 0.008 ).float() - 1.0) * (-1)
        mask = mask.type_as( dist_without_min_on_diag ) * 10
        dist_without_min_on_diag = dist_without_min_on_diag + mask
        if self.batch_reduce == 'min':
            min_neg = torch.min( dist_without_min_on_diag, 1 )[0]
            if self.anchor_swap:
                min_neg2 = torch.min( dist_without_min_on_diag, 0 )[0]
                min_neg = torch.min( min_neg, min_neg2 )
            pos= pos1
        else:
            print( 'Unknown batch reduce mode. Try min' )
            sys.exit( 1 )
        if self.loss_type == "triplet_margin":
            loss = torch.clamp( self.margin + pos - min_neg, min=0.0 )
        elif self.loss_type == 'softmax':
            exp_pos = torch.exp( 2.0 - pos )
            exp_den = exp_pos + torch.exp( 2.0 - min_neg ) + self.eps
            loss = - torch.log( exp_pos / exp_den )
        elif self.loss_type == 'contrastive':
            loss = torch.clamp( self.margin - min_neg, min=0.0 ) + pos
        else:
            print( 'Unknown loss type. Try triplet_margin, softmax or contrastive' )
            sys.exit( 1 )
        loss = torch.mean( loss )
        return loss
