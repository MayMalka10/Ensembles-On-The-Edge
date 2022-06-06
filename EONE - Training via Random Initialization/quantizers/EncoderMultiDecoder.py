from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from quantizers import EncoderDecoder


class EncoderMultiDecoder( EncoderDecoder.EncoderDecoder ):
    def __init__(self, encoder, decoder, primary_loss, **kwargs):
        super().__init__(encoder, decoder, primary_loss, **kwargs)
        self.decoder = nn.ModuleList(self.decoder)

    def decode(self, z):
        predictions = []
        for decoder in self.decoder:
            predictions.append( decoder( z ) )
        return predictions

    def calculate_prime_loss(self, y_hat_list, y):
        loss = 0
        for y_hat in y_hat_list:
            loss += self.primary_loss(y_hat, y)
        return loss / len(y_hat_list)

    def calculate_acc(self, preds_list, gts, handlers, activation):
        accs = []
        ensemble_preds = self.ensemble_calculator(preds_list)
        for preds, handler in zip(preds_list, handlers[0:len(preds_list)]):
            accs.append(handler(activation(preds), gts))
        accs.append(handlers[-1](activation(ensemble_preds), gts))
        return accs
