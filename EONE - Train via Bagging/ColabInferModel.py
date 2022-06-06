import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from User1ModelFunctions import FirstUserFunctions
from RestModelFunctions import RestUsersFunctions

class NeuraQuantModel(FirstUserFunctions):
    def __init__(self, encoder, decoder, primary_loss, n_embed=1024,n_parts=1, decay=0.8, commitment=1., eps=1e-5,
                 skip_quant=False, learning_rate=1e-3):
        super().__init__(encoder, decoder, primary_loss,n_embed,n_parts,decay, commitment, eps,
                 skip_quant, learning_rate)
        self.decoder = nn.ModuleList(self.decoder)

    def process_batch(self, batch):
        commit_loss = 0
        x, y = batch
        z_e = self.encode(x)
        z_q, indices, commit_loss = self.quantize(z_e)
        z_q = z_q.view((z_q.shape[0], z_q.shape[3], z_q.shape[1], z_q.shape[2]))
        y_hat = self.decode(z_q)
        ensemble_y_hat = self.ensemble_calculator(y_hat)
        batch_acc, batch_acc_ensemble = self.accuracy(y, y_hat, ensemble_y_hat)
        prime_loss = self.calculate_prime_loss(y_hat,y)
        result_dict = {'loss': prime_loss + commit_loss, 'preds': y_hat, 'gts': y}
        return result_dict, batch_acc, y_hat






class NeuraQuantModel2(RestUsersFunctions):
    def __init__(self, encoder, decoder, quantizer, primary_loss, n_embed=1024,n_parts=1, decay=0.8, commitment=1., eps=1e-5,
                 skip_quant=True, learning_rate=1e-3):
        super().__init__(encoder, decoder, primary_loss,n_embed,n_parts,decay, commitment, eps,
                 skip_quant, learning_rate)
        self.encoder = encoder
        self.decoder = nn.ModuleList(self.decoder)
        self.quantizer = quantizer

    def process_batch_fixed(self,batch):
        x, y = batch
        z_e = self.encoder(x)
        if not self.skip_quant:
            z_e = z_e.view((z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1]))
            z_q, indices, commit_loss = self.quantizer(z_e)
            z_e = z_q.view((z_q.shape[0], z_q.shape[3], z_q.shape[1], z_q.shape[2]))
        y_hat = self.decode(z_e)
        ensemble_y_hat = self.ensemble_calculator(y_hat)
        batch_acc, batch_acc_ensemble = self.accuracy(y, y_hat, ensemble_y_hat)
        prime_loss = self.calculate_prime_loss(y_hat, y)
        result_dict = {'loss': prime_loss, 'preds': y_hat, 'gts': y}
        return result_dict, batch_acc, y_hat