import os
import torch
from torchvision import datasets


def get_mnist(train_size, validation_size, test_size, pool_size, device):
    data_dir = "mnist"
    os.makedirs(data_dir, exist_ok=True)
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train_all = train_ds.data
    y_train_all = train_ds.targets
    x_test_all = test_ds.data
    y_test_all = test_ds.targets

    # We will take "train_size + validation_size + pool_size" samples from the training
    # set and "test_size" samples from the test set.

    # Taking test indices is easy. In principle, I should always have test_size = 10k,
    # But I allow taking more since I need this for faster testing during development.
    test_indices = torch.randperm(x_test_all.shape[0])[:test_size]
    
    # Choose class-balanced samples for the initial training set
    per_class = train_size // 10
    train_class_indices = []
    for label in range(10):
        class_indices = torch.where(y_train_all == label)[0]
        chosen = class_indices[torch.randperm(class_indices.numel())[:per_class]]
        train_class_indices.append(chosen)
    train_indices = torch.cat(train_class_indices)
    train_indices = train_indices[torch.randperm(train_indices.numel())]
    remaining_mask = torch.ones(x_train_all.shape[0], dtype=torch.bool)
    remaining_mask[train_indices] = False
    remaining_indices = torch.where(remaining_mask)[0]
    
    # Remaining indices will be for validation and pool sets
    val_pool_size = validation_size + pool_size
    val_pool_indices = remaining_indices[torch.randperm(remaining_indices.numel())[:val_pool_size]]

    train = slice_to_tensor(x_train_all, y_train_all, train_indices, device)
    validation = slice_to_tensor(x_train_all, y_train_all, val_pool_indices[:validation_size], device)
    pool = slice_to_tensor(x_train_all, y_train_all, val_pool_indices[validation_size:], device)
    test = slice_to_tensor(x_test_all, y_test_all, test_indices[:test_size], device)

    return train, validation, pool, test

def slice_to_tensor(x_src, y_src, indices, device):
    x = x_src.index_select(0, indices).to(torch.float32) / 255.0
    y = y_src.index_select(0, indices).to(torch.int64)
    return x.to(device), y.to(device)
