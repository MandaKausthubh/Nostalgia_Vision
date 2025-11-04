from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _norm_from_model_or_default(model_id: Optional[str]):
    if model_id:
        proc = AutoImageProcessor.from_pretrained(model_id)
        mean, std = proc.image_mean, proc.image_std
        size = proc.size.get("shortest_edge", 224)
    else:
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD
        size = 224
    return mean, std, size


def get_cifar100(root: str, model_id: str, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    mean, std, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=mean, std=std)
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_cifar10(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    mean, std, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=mean, std=std)
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_caltech256(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    mean, std, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=mean, std=std)
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    # Caltech256 does not have official train/test splits; common practice is custom split.
    # Here we use entire dataset as train for loader build smoke-test; user should create splits externally.
    ds = datasets.Caltech256(root=root, download=True, transform=train_tf)
    n = len(ds)
    n_train = int(0.8 * n)
    train, test = torch.utils.data.random_split(ds, [n_train, n - n_train])
    # Override transform for test subset
    test.dataset.transform = test_tf

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def _get_imagefolder_generic(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """
    Generic ImageFolder loader assuming `root/train` and `root/val` subdirectories.
    """
    mean, std, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=mean, std=std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.ImageFolder(root=f"{root}/train", transform=train_tf)
    val = datasets.ImageFolder(root=f"{root}/val", transform=val_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_imagenet1k(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    mean, std, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=mean, std=std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])
    train = datasets.ImageFolder(root=f"{root}/train", transform=train_tf)
    val = datasets.ImageFolder(root=f"{root}/val", transform=val_tf)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# Wrappers for generic ImageFolder datasets used in sequences

def get_tinyimagenet(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_imagenet100(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_imagenet_r(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_imagenet_a(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)
