"""
dataset.py - Data loading utilities for diffusion models

Contains:
- get_fashion_mnist_loaders: Load and split FashionMNIST dataset
- get_mnist_loaders: Load and split MNIST dataset
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transform():
    """
    Get the standard transform for diffusion models.
    
    Transforms images to [-1, 1] range which is standard for diffusion models.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # [-1, 1]
    ])


def get_fashion_mnist_loaders(
    data_dir='./data',
    batch_size=128,
    train_ratio=0.9,
    num_workers=0,
    download=True
):
    """
    Load FashionMNIST dataset with train/validation split.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of training data (rest is validation)
        num_workers: Number of workers for DataLoader
        download: Whether to download if not present
    
    Returns:
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        full_dataset: The full dataset (for reference)
    """
    transform = get_transform()
    
    # Load full training dataset
    full_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=True, 
        download=download, 
        transform=transform
    )
    
    # Split into train and validation
    train_size = int(train_ratio * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=num_workers
    )
    
    print(f"FashionMNIST loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(valid_loader)}")
    
    return train_loader, valid_loader, full_dataset


def get_mnist_loaders(
    data_dir='./data',
    batch_size=128,
    train_ratio=0.9,
    num_workers=0,
    download=True
):
    """
    Load MNIST dataset with train/validation split.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of training data (rest is validation)
        num_workers: Number of workers for DataLoader
        download: Whether to download if not present
    
    Returns:
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        full_dataset: The full dataset (for reference)
    """
    transform = get_transform()
    
    # Load full training dataset
    full_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=download, 
        transform=transform
    )
    
    # Split into train and validation
    train_size = int(train_ratio * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=num_workers
    )
    
    print(f"MNIST loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(valid_loader)}")
    
    return train_loader, valid_loader, full_dataset


if __name__ == "__main__":
    # Test data loading
    train_loader, valid_loader, _ = get_fashion_mnist_loaders(batch_size=64)
    
    # Check a batch
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"\nBatch shape: {images.shape}")
    print(f"Value range: [{images.min():.2f}, {images.max():.2f}]")

