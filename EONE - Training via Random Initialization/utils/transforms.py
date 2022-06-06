import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import kornia
import PIL
import torch
import numpy as np

class PILToTensor:
    """Convert a ``PIL Image`` to a tensor of the same type. This transform does not support torchscript.

    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        tensor = pil_to_tensor(pic)
        tensor[tensor == 255] = 0
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'

def pil_to_tensor(pic):
    """Convert a ``PIL Image`` to a tensor of the same type.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    # handle PIL Image
    img = torch.as_tensor(np.array(pic))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img

def im_to_float(im):
    return im.float().div( 255 )

class TensorDatasetWithTransform(TensorDataset):
    def __init__(self, data, labels, transforms):
        super().__init__(data, labels)
        self.transforms = transforms

    def __getitem__(self, item):
        example, label = super().__getitem__(item)
        return self.transforms(example), label
def get_cifar_train_transforms():
    transform = transforms.Compose(
                [transforms.RandomResizedCrop( (32, 32), scale=(0.8, 0.8) ),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return transform
def get_imagenet_train_transforms():
    transform = transforms.Compose(
                [transforms.RandomResizedCrop( (224, 224), scale=(0.8, 0.8) ),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return transform
def get_test_transform():
    test_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return test_transform

def get_test_torch_transform():
    test_torch_transform = transforms.Compose(
        [im_to_float,
         transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    return test_torch_transform

def get_img_and_mask_transforms(desired_size):
    img_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(desired_size),
                 transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )] )
    tgt_transform = transforms.Compose(
                [PILToTensor(),
                 transforms.Resize(desired_size, interpolation=PIL.Image.NEAREST),
                 lambda x: x.long()]
    )
    return img_transform, tgt_transform
