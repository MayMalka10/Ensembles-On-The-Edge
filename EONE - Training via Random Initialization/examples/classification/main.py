import os
from utils.transforms import get_cifar_train_transforms, get_test_transform, get_imagenet_train_transforms
import torchvision
import torch
from architectures.mobilenets import get_mobilenet_parts
from examples.classification.modules import encoderMultiClassifier
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import hydra
from omegaconf import DictConfig
import sys
from PIL import Image
from architectures import utils
import re
from utils.file_handling import strip_prefix

is_debug = sys.gettrace() is not None


def imagenet_loader(path: str) -> Image.Image:
    with open( path, 'rb' ) as f:
        img = Image.open( f )
        img = img.resize( (224, 224) )
        return img.convert( 'RGB' )


@hydra.main( config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    log_dir_name = f"{cfg.dataset.dataset.name}_{cfg.arch.arch}_{cfg.quantization.quantization}_{cfg.ensemble.ensemble}".replace(
        ":", '-' ).replace( '\'', '' ).replace( ' ', '' )
    log_dir_name = log_dir_name.replace( '{', '' )
    log_dir_name = log_dir_name.replace( '}', '' )
    log_dir_name = log_dir_name.replace( ',', '_' )
    log_dir = os.path.join( os.getcwd(), log_dir_name )
    if is_debug:
        print( 'in debug mode!' )
        training_params = cfg.training.training_debug
    else:
        print( 'in run mode!' )
        training_params = cfg.training.training
    pre_trained_path = training_params.pre_trained_path
    num_epochs = training_params.num_epochs
    loss = torch.nn.CrossEntropyLoss()
    test_transform = get_test_transform()
    if 'cifar' in cfg.dataset.dataset['name']:
        train_transform = get_cifar_train_transforms()
    elif cfg.dataset.dataset['name'] in ['imagenet', 'imagenette', 'imagewoof']:
        train_transform = get_imagenet_train_transforms()
    else:
        train_transform = test_transform

    if cfg.dataset.dataset['name'] == 'cifar10':
        trainset = torchvision.datasets.CIFAR10( root=cfg.params.data_path, train=True,
                                                 download=True, transform=train_transform )
        testset = torchvision.datasets.CIFAR10( root=cfg.params.data_path, train=False,
                                                download=True, transform=test_transform )
    elif cfg.dataset.dataset['name'] == 'cifar100':
        trainset = torchvision.datasets.CIFAR100( root=cfg.params.data_path, train=True,
                                                  download=True, transform=train_transform )
        testset = torchvision.datasets.CIFAR100( root=cfg.params.data_path, train=False,
                                                 download=True, transform=test_transform )
    elif cfg.dataset.dataset['name'] == 'imagenet':
        trainset = torchvision.datasets.ImageNet( root=cfg.params.data_path, train=True,
                                                  download=True, transform=train_transform )
        testset = torchvision.datasets.ImageNet( root=cfg.params.data_path, train=False,
                                                 download=True, transform=test_transform )
    elif cfg.dataset.dataset['name'] == 'caltech-256':
        trainset = torchvision.datasets.Caltech256( root=cfg.params.data_path, train=True,
                                                    download=True, transform=train_transform )
        testset = torchvision.datasets.Caltech256( root=cfg.params.data_path, train=False,
                                                   download=True, transform=test_transform )
    elif cfg.dataset.dataset['name'] == 'imagenette':
        trainset = torchvision.datasets.ImageFolder( root=os.path.join(cfg.params.data_path, 'imagenette2-320/train'),
                                                     transform=train_transform, loader=imagenet_loader)
        testset = torchvision.datasets.ImageFolder( root=os.path.join(cfg.params.data_path, 'imagenette2-320/val'),
                                                    transform=test_transform, loader=imagenet_loader)
    elif cfg.dataset.dataset['name'] == 'imagewoof':
        trainset = torchvision.datasets.ImageFolder( root=os.path.join(cfg.params.data_path, 'imagewoof2-320/train'),
                                                     transform=train_transform, loader=imagenet_loader)
        testset = torchvision.datasets.ImageFolder( root=os.path.join(cfg.params.data_path, 'imagewoof2-320/val'),
                                                    transform=test_transform, loader=imagenet_loader)


    trainloader = torch.utils.data.DataLoader( trainset, batch_size=training_params.batch_size,
                                               shuffle=True, num_workers=training_params.num_workers)
    testloader = torch.utils.data.DataLoader( testset, batch_size=training_params.batch_size,
                                              shuffle=False, num_workers=training_params.num_workers)
    if cfg.arch.arch.name == 'mobilenet':
        use_transfer_learning = cfg.arch.arch.pretrained
        if use_transfer_learning:
            weight_reset = utils.add_noise_to_model
        else:
            weight_reset = utils.weight_reset
        parts_dict = get_mobilenet_parts(pretrained=use_transfer_learning,
                                      weight_reset=weight_reset)
    # load encoder and decoders params (in case our checkpoint is not fully compatible with requested setup)
    if pre_trained_path is not None:
        checkpoint = torch.load( pre_trained_path )
        state_dict = strip_prefix( checkpoint['state_dict'], prefix='encoder')
        encoder = parts_dict['encoder']
        encoder.load_state_dict( state_dict )
        n_ensemble_pre_trained = int(re.search(r'\d+', pre_trained_path.split('n_ensemble-')[1]).group())
        for decoder_idx in range(min(n_ensemble_pre_trained, cfg.ensemble.ensemble.n_ensemble)):
            state_dict = strip_prefix( checkpoint['state_dict'], prefix=f"decoder.{decoder_idx}")
            parts_dict['decoders'][decoder_idx].load_state_dict( state_dict )

    module = encoderMultiClassifier.EncoderMultiClassifier( encoder=parts_dict['encoder'],
                                                            decoder=parts_dict['decoders'],
                                                            primary_loss=loss,)
    tb_logger = pl_loggers.TensorBoardLogger( log_dir )
    if pre_trained_path is not None:
        n_embed_pre_trained = int( re.search( r'\d+', pre_trained_path.split( 'n_embed-' )[1] ).group() )
        if n_embed_pre_trained != cfg.quantization.quantization.n_embed:
            pre_trained_path = None# invalidate the path, since it's not fully compatible
    trainer = pl.Trainer( max_epochs=num_epochs, gpus=1, logger=tb_logger, resume_from_checkpoint=pre_trained_path)
    trainer.fit( module, trainloader, testloader )


if __name__ == '__main__':
    train()
