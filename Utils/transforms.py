import torchvision.transforms as transforms


def get_test_transform():
    test_transform = transforms.Compose(
                [transforms.ToTensor(),
      		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return test_transform

def get_cifar_train_transforms():
    transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return transform

