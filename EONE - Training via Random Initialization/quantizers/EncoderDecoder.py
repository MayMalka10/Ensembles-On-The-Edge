from torch import nn
import torch
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
import pytorch_lightning as pl
import hydra

class EncoderDecoder(pl.LightningModule):
    def __init__(self, encoder, decoder, primary_loss, decay=0.8, eps=1e-5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decay = decay
        self.eps = eps
        self.primary_loss = primary_loss
        cfg = hydra.compose(config_name="config")
        self.skip_quant = cfg.quantization.quantization['skip']
        self.n_embed = cfg.quantization.quantization['n_embed']
        self.n_parts = cfg.quantization.quantization['n_parts']
        self.commitment_w = cfg.quantization.quantization['commitment_w']
        self.learning_rate = cfg.training.training['lr']
        self.hparams.update(dict(n_embed=1024, decay=0.8, commitment_w=1., eps=1e-5))
        dummy_input = torch.zeros((1, 3, 100, 100), device=self.device)
        self.quant_dim = encoder(dummy_input).shape[1]
        self.quantizer = VectorQuantize(
            dim=self.quant_dim // self.n_parts,
            codebook_size=self.n_embed,     # size of the dictionary
            decay=self.decay,       # the exponential moving average decay, lower means the dictionary will change faster
            commitment=1.0    # the weight on the commitment loss (==1 cause we want control)
        )


    def encode(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view((z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1]))
        if not self.skip_quant:
            z_e_split = torch.split(z_e, self.quant_dim // self.n_parts, dim=3)
            z_q_split, indices_split = [], []
            commit_loss = 0
            for z_e_part in z_e_split:
                z_q_part, indices_part, commit_loss_part = self.quantizer(z_e_part)
                commit_loss += commit_loss_part
                z_q_split.append(z_q_part)
                indices_split.append( indices_part )
            z_q = torch.cat(z_q_split, dim=3)
            indices = torch.stack(indices_split, dim=3)
        else:
            z_q, indices, commit_loss = z_e, None, 0
        return z_q, indices, commit_loss

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_q, _, _ = self.encode(x)
        z_q = z_q.view((z_q.shape[0], z_q.shape[3], z_q.shape[1], z_q.shape[2]))
        return self.decode(z_q)

    def calculate_prime_loss(self, y_hat, y):
        return self.primary_loss(y_hat, y)


    def process_batch(self, batch):
        x, y = batch
        z_q, indices, commit_loss = self.encode( x )
        z_q = z_q.view( (z_q.shape[0], z_q.shape[3], z_q.shape[1], z_q.shape[2]) )
        y_hat = self.decode( z_q )
        prime_loss = self.calculate_prime_loss( y_hat, y )
        return prime_loss, commit_loss, y_hat, y

    def training_step(self, batch, batch_idx):
        prime_loss, commit_loss, preds, gts = self.process_batch(batch)
        self.log('prime_loss', prime_loss)
        self.log('commit_loss', commit_loss)
        return {'loss': prime_loss + commit_loss * self.commitment_w, 'preds': preds, 'gts': gts}

    def training_epoch_end(self, outputs_dicts) -> None:
        loss_sum = 0
        for output in outputs_dicts:
            loss_sum += output['loss']
        self.log('train_loss_epoch', loss_sum / len(outputs_dicts))

    def validation_step(self, batch, batch_idx):
        prime_loss, commit_loss, preds, gts = self.process_batch( batch )
        self.log( 'val_prime_loss', prime_loss )
        self.log( 'val_commit_loss', commit_loss )
        return {'loss': prime_loss + commit_loss * self.commitment_w, 'preds': preds, 'gts': gts}

    def validation_epoch_end(self, outputs_dicts) -> None:
        loss_sum = 0
        for output in outputs_dicts:
            loss_sum += output['loss']
        self.log( 'val_loss_epoch', loss_sum / len( outputs_dicts ) )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20)
        return optimizer#, scheduler
