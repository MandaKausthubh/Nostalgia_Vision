from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import AutoImageProcessor

try:
    from datasets import load_dataset, Dataset as HFDataset
except Exception:  # datasets may not be installed yet
    load_dataset = None
    HFDataset = None


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


# --- Torchvision helpers (retain old APIs) ---

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
    ds = datasets.Caltech256(root=root, download=True, transform=train_tf)
    n = len(ds)
    n_train = int(0.8 * n)
    train, test = torch.utils.data.random_split(ds, [n_train, n - n_train])
    test.dataset.transform = test_tf

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def _get_imagefolder_generic(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
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


def get_tinyimagenet(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_imagenet100(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_imagenet_r(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_imagenet_a(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    return _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)


def get_mnist(root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    # For MNIST, upsample to match model input size
    _, _, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
    # Convert grayscale to 3 channels
    to3 = transforms.Lambda(lambda x: x.expand(3, x.shape[1], x.shape[2]))
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        to3,
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        to3,
        normalize,
    ])
    train = datasets.MNIST(root=root, train=True, download=True, transform=train_tf)
    test = datasets.MNIST(root=root, train=False, download=True, transform=test_tf)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True), \
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# --- HF-first unified loader ---

_HF_NAME_MAP: Dict[str, str] = {
    'cifar10': 'cifar10',
    'cifar100': 'cifar100',
    'tinyimagenet': 'Maysee/tiny-imagenet',
    'caltech256': 'caltech256',
    'cub200': 'caltech_birds2011',
    'flowers102': 'oxford_flowers102',
    'mnist': 'mnist',
    # Robustness sets below may not be directly hosted; fallback will handle
    'imagenet100': 'imagenet-100',
    'imagenet-r': 'imagenet-r',
    'imagenet-a': 'imagenet-a',
}


class HFDatasetTorch(Dataset):
    def __init__(self, hf_ds, image_key: str, label_key: str, tfm, classes: List[str]):
        self.ds = hf_ds
        self.image_key = image_key
        self.label_key = label_key
        self.tfm = tfm
        self.classes = classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex[self.image_key]
        lbl = ex[self.label_key]
        # Some HF datasets return dicts for images
        from PIL import Image
        if not isinstance(img, Image.Image):
            if isinstance(img, dict) and 'path' in img:
                img = Image.open(img['path']).convert('RGB')
            else:
                # Assume numpy array
                import numpy as np
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
        x = self.tfm(img)
        return x, int(lbl)


def _hf_try_build(name: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int, root: Optional[str] = None):
    if load_dataset is None:
        raise RuntimeError("datasets library not available")
    repo = _HF_NAME_MAP.get(name)
    if repo is None:
        raise RuntimeError("no HF mapping")
    try:
        ds = load_dataset(repo)
    except Exception as e:
        raise RuntimeError(f"HF load failed: {e}")

    # Determine splits
    split_keys = list(ds.keys())
    # Common patterns
    if {'train', 'validation', 'test'}.issubset(split_keys):
        train_split = ds['train'].shuffle(seed=0).flatten_indices()
        val_split = ds['test']
        # Optionally merge validation into train
        train_split = datasets.ConcatDataset([train_split, ds['validation']]) if False else train_split  # placeholder
    elif {'train', 'valid'}.issubset(split_keys):
        train_split, val_split = ds['train'], ds['valid']
    elif {'train', 'val'}.issubset(split_keys):
        train_split, val_split = ds['train'], ds['val']
    elif {'train', 'test'}.issubset(split_keys):
        train_split, val_split = ds['train'], ds['test']
    else:
        # Fallback to arbitrary 80/20 split on the first available split
        base = ds[split_keys[0]]
        n = len(base)
        n_train = int(0.8 * n)
        train_split = base.select(range(n_train))
        val_split = base.select(range(n_train, n))

    # Identify keys
    feat = train_split.features
    image_key = 'image' if 'image' in feat else next((k for k, v in feat.items() if getattr(v, '_type', '') == 'Image'), 'image')
    label_key = 'label' if 'label' in feat else next((k for k, v in feat.items() if v.__class__.__name__ == 'ClassLabel'), 'label')

    # Classes
    classes = []
    try:
        classes = feat[label_key].names  # type: ignore
    except Exception:
        # try to infer
        import numpy as np
        mx = int(max(train_split[label_key]))
        classes = [str(i) for i in range(mx + 1)]

    mean, std, size_from_model = _norm_from_model_or_default(model_id)
    size = image_size or size_from_model
    normalize = transforms.Normalize(mean=mean, std=std)
    common = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    aug = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_torch = HFDatasetTorch(train_split, image_key, label_key, aug, classes)
    val_torch = HFDatasetTorch(val_split, image_key, label_key, common, classes)

    train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_torch, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(classes)


def get_loaders_and_num_classes(name: str, root: str, model_id: Optional[str], image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, int]:
    """Unified entry that tries HF first (if mapped), then torchvision/local fallbacks.
    Returns train_loader, val_loader, num_classes.
    """
    # HF first
    try:
        train_loader, val_loader, ncls = _hf_try_build(name, model_id, image_size, batch_size, num_workers, root)
        return train_loader, val_loader, ncls
    except Exception:
        pass

    # Torchvision fallbacks
    name_l = name.lower()
    if name_l == 'cifar100':
        tr, va = get_cifar100(root, model_id, image_size, batch_size, num_workers)
        return tr, va, 100
    if name_l == 'cifar10':
        tr, va = get_cifar10(root, model_id, image_size, batch_size, num_workers)
        return tr, va, 10
    if name_l == 'caltech256':
        tr, va = get_caltech256(root, model_id, image_size, batch_size, num_workers)
        # torchvision Caltech256 has 256 categories
        return tr, va, 256
    if name_l in ('tinyimagenet', 'imagenet100', 'imagenet-r', 'imagenet-a'):
        tr, va = _get_imagefolder_generic(root, model_id, image_size, batch_size, num_workers)
        ncls = getattr(tr.dataset, 'classes', None)
        n = len(ncls) if ncls is not None else 1000
        return tr, va, n
    if name_l == 'mnist':
        tr, va = get_mnist(root, model_id, image_size, batch_size, num_workers)
        return tr, va, 10

    raise ValueError(f"Unknown dataset for loaders: {name}")
