import os
from utils.transforms import get_img_and_mask_transforms
import torchvision
import torch
from examples.segmentation.modules import encoderMultiSegmentor
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import hydra
from omegaconf import DictConfig
import sys
is_debug = sys.gettrace() is not None
from architectures import seg_nets
from kornia.losses import DiceLoss
from segmentation_models_pytorch.utils import losses
from torch import nn


class CrossEntropyPlusDiceLoss( nn.Module ):
    def __init__(self, w):
        super().__init__()
        self.ce_loss = losses.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.w = w
    def forward(self, y_hat, y):
        return self.ce_loss(y_hat, y.squeeze(1)) + self.w * self.dice_loss(y_hat, y.squeeze(1))


@hydra.main( config_path='conf/config.yaml' )
def train(cfg: DictConfig) -> None:
    log_dir_name = f"{cfg.dataset.name}_{cfg.arch.name}_{cfg.quantization}_{cfg.ensemble}".replace(":",'-').replace('\'','').replace(' ', '')
    log_dir_name = log_dir_name.replace('{','')
    log_dir_name = log_dir_name.replace( '}', '')
    log_dir_name = log_dir_name.replace( ',', '_')
    log_dir = os.path.join(os.getcwd(), log_dir_name)
    pre_trained_path = None
    if is_debug:
        print( 'in debug mode!')
        training_params = cfg.training_debug
    else:
        print('in run mode!')
        training_params = cfg.training

    loss = CrossEntropyPlusDiceLoss( w=0.0 )#losses.CrossEntropyLoss() + losses.DiceLoss()
    img_transform, tgt_transform = get_img_and_mask_transforms((256, 256))



    if cfg.arch.name == "psp":
        parts_dict = seg_nets.get_psp_parts( pretrained=True, num_classes=cfg.dataset.num_classes,
                                             decoder_copies=cfg.ensemble.n_ensemble, encoder_name=cfg.arch.backbone)

    module = encoderMultiSegmentor.EncoderMultiSegmentor(encoder=parts_dict['encoder'], decoder=parts_dict['decoders'],
                                                           primary_loss=loss, n_classes=cfg.dataset.num_classes,
                                                         n_embed=cfg.quantization.n_embed,
                                                         commitment=cfg.quantization.commitment_w, skip_quant=True,
                                                         learning_rate=training_params.lr)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    trainer = pl.Trainer (max_epochs=100, gpus=1, logger=tb_logger)
    if cfg.dataset['name'] == 'voc':
        trainset = torchvision.datasets.VOCSegmentation( root=cfg.params.data_path, image_set="train",
                                                 download=False, transform=img_transform, target_transform=tgt_transform, year=cfg.dataset['year'])
        testset = torchvision.datasets.VOCSegmentation( root=cfg.params.data_path, image_set="val",
                                                download=False, transform=img_transform, target_transform=tgt_transform, year=cfg.dataset['year'])
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=training_params.batch_size,
                                               shuffle=True, num_workers=training_params.num_workers )
    testloader = torch.utils.data.DataLoader( testset, batch_size=training_params.batch_size,
                                              shuffle=False, num_workers=training_params.num_workers )

    trainer.fit(module, trainloader, testloader)

if __name__ == '__main__':
    train()