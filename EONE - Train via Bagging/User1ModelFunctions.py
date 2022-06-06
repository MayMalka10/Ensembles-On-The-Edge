import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
# from Quantizer import VQVAE
import ColabInferModel


class FirstUserFunctions(nn.Module):
    def __init__(self, encoder, decoder, primary_loss, n_embed,n_parts, decay=0.8, commitment=1., eps=1e-5,
                 skip_quant=False, learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.primary_loss = primary_loss
        self.commitment_w = commitment
        dummy_input = torch.zeros((1, 3, 100, 100)) # Check number of channels the encoder outputs
        self.quant_dim = encoder(dummy_input).shape[1]
        self.n_parts = n_parts
        self.quantizer = VectorQuantize(dim=self.quant_dim // self.n_parts,
                                        codebook_size=self.n_embed,  # size of the dictionary
                                        decay=self.decay,
                                        # the exponential moving average decay, lower means the dictionary will change faster
                                        commitment=1.0)  # the weight on the commitment loss (==1 cause we want control))
        self.skip_quant = skip_quant
        self.learning_rate = learning_rate


    def encode(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view((z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1]))
        return z_e

    def quantize(self,z_e):
        if not self.skip_quant:
            z_e_split = torch.split(z_e, self.quant_dim // self.n_parts, dim=3)
            z_q_split, indices_split = [], []
            commit_loss = 0
            for z_e_part in z_e_split:
                z_q_part, indices_part, commit_loss_part = self.quantizer(z_e_part)
                commit_loss += commit_loss_part
                z_q_split.append(z_q_part)
                indices_split.append(indices_part)
            z_q = torch.cat(z_q_split, dim=3)
            indices = torch.stack(indices_split, dim=3)
        else:
            z_q, indices, commit_loss = z_e, None, 0
        return z_q, indices, commit_loss


    def decode(self, z):
        predictions = []
        for decoder in self.decoder:
            predictions.append(decoder(z))
        return predictions

    def calculate_prime_loss(self, y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += self.primary_loss(y_hat, y)
        return loss / len(y_hat_list)

    def ensemble_calculator(self, preds_list):
        return torch.mean(torch.stack(preds_list), axis=0 )

    def accuracy(self,y,y_pred,ensemble_y_pred):
        ens_pred = torch.max(ensemble_y_pred.data, 1)[1]
        batch_ens_corr = (ens_pred == y).sum()
        predicted = []
        batch_corr = []
        for vec in range(len(y_pred)):
            predicted.append(torch.max(y_pred[vec].data, 1)[1])
            batch_corr.append((predicted[vec] == y).sum())
        return batch_corr, batch_ens_corr


