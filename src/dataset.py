import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_transforms(image_size=(32, 32)):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])
    
    return transform_train, transform_test


def get_dataloaders(batch_size=64, val_split=0.1, test_split=0.1, root="data", subset_classes=None):
    transform_train, transform_test = get_transforms()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, root)

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_full = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    if subset_classes is not None:
        class_to_idx = {cls: i for i, cls in enumerate(train_full.classes)}
        subset_idx = [i for i, (_, label) in enumerate(train_full) if label in [class_to_idx[c] for c in subset_classes]]
        train_full = torch.utils.data.Subset(train_full, subset_idx)

    class_names = train_full.classes

    total_len = len(train_full)
    val_len = int(val_split * total_len)
    train_len = total_len - val_len
    train_set, val_set = random_split(train_full, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names
