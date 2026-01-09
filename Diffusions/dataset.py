"""
dataset.py - Data loading utilities for diffusion models

Contains:
- get_fashion_mnist_loaders: Load and split FashionMNIST dataset
- get_mnist_loaders: Load and split MNIST dataset
- get_oxford_flowers_loaders: Load Oxford Flowers dataset
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transform(image_size=None):
    """
    Get the standard transform for diffusion models.
    
    Transforms images to [-1, 1] range which is standard for diffusion models.
    Uses resize(1.5x) + centerCrop for better aspect ratio handling and content preservation.
    
    Args:
        image_size: If provided, resize images to this size (H, W). 
                   If None, no resizing is applied.
    """
    transform_list = []
    
    if image_size is not None:
        # Resize to 1.5x then center crop for better aspect ratio handling
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        # Resize to 1.5x the target size to preserve more content
        resize_size = (int(image_size[0] * 1.5), int(image_size[1] * 1.5))
        transform_list.append(transforms.Resize(resize_size))
        transform_list.append(transforms.CenterCrop(image_size))
    
    transform_list.extend([
        transforms.ToTensor(),  # [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # [-1, 1]
    ])
    
    return transforms.Compose(transform_list)


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


def get_oxford_flowers_loaders(
    data_dir='./data',
    batch_size=32,
    image_size=64,
    num_workers=0,
    download=True,
    use_builtin_splits=True
):
    """
    Load Oxford Flowers-102 dataset.
    
    The Oxford Flowers dataset has built-in train/val/test splits.
    By default, uses train split for training and val split for validation.
    If use_builtin_splits=False, combines train+val and splits manually.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size for DataLoader (smaller than MNIST due to larger images)
        image_size: Size to resize images to (default: 64x64)
        num_workers: Number of workers for DataLoader
        download: Whether to download if not present
        use_builtin_splits: If True, use built-in train/val splits. 
                           If False, combine train+val and split manually.
    
    Returns:
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        full_dataset: Dictionary with train/val/test datasets
    """
    transform = get_transform(image_size=image_size)
    
    if use_builtin_splits:
        # Use built-in splits: train for training, val for validation
        train_dataset = datasets.Flowers102(
            root=data_dir,
            split='train',
            download=download,
            transform=transform
        )
        valid_dataset = datasets.Flowers102(
            root=data_dir,
            split='val',
            download=download,
            transform=transform
        )
        test_dataset = datasets.Flowers102(
            root=data_dir,
            split='test',
            download=download,
            transform=transform
        )
    else:
        # Combine train+val and split manually
        train_val_dataset = datasets.Flowers102(
            root=data_dir,
            split='train',
            download=download,
            transform=transform
        )
        val_dataset = datasets.Flowers102(
            root=data_dir,
            split='val',
            download=download,
            transform=transform
        )
        
        # Combine train and val datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([train_val_dataset, val_dataset])
        
        # Split 90/10
        train_size = int(0.9 * len(combined_dataset))
        valid_size = len(combined_dataset) - train_size
        train_dataset, valid_dataset = random_split(combined_dataset, [train_size, valid_size])
        
        test_dataset = datasets.Flowers102(
            root=data_dir,
            split='test',
            download=download,
            transform=transform
        )
    
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
    
    print(f"Oxford Flowers-102 loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(valid_dataset)}")
    if use_builtin_splits:
        print(f"  Test samples: {len(test_dataset)}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(valid_loader)}")
    
    full_dataset = {
        'train': train_dataset,
        'val': valid_dataset,
        'test': test_dataset if use_builtin_splits else None
    }
    
    return train_loader, valid_loader, full_dataset


if __name__ == "__main__":
    # Test data loading
    print("Testing FashionMNIST...")
    train_loader, valid_loader, _ = get_fashion_mnist_loaders(batch_size=64)
    
    # Check a batch
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print(f"Value range: [{images.min():.2f}, {images.max():.2f}]\n")
    
    # Test Oxford Flowers
    print("Testing Oxford Flowers-102...")
    try:
        train_loader_flowers, valid_loader_flowers, _ = get_oxford_flowers_loaders(
            batch_size=16, 
            image_size=64
        )
        batch_flowers = next(iter(train_loader_flowers))
        images_flowers, labels_flowers = batch_flowers
        print(f"Batch shape: {images_flowers.shape}")
        print(f"Value range: [{images_flowers.min():.2f}, {images_flowers.max():.2f}]")
    except Exception as e:
        print(f"Could not load Oxford Flowers dataset: {e}")
        print("Note: You may need to download it first by running with download=True")

