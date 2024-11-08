"""
This module provides utility functions for downloading and preparing datasets 
for machine learning tasks, specifically for image classification tasks.

Functions:
    - download_data: Downloads and extracts a dataset of pizza, steak, and sushi images 
                     if not already present in the './data/' directory. 
                     It ensures the dataset is ready for use.
                     
    - get_dataloaders: Loads training and testing datasets from specified directories,
                        applies transformations, and returns PyTorch DataLoaders along with 
                        class labels from the training dataset.
    
Modules:
    - requests: Used for downloading data from a URL.
    - zipfile: Used for extracting the contents of the downloaded ZIP file.
    - torchvision: Used for loading image datasets and applying transformations.
    - torch.utils.data: Provides DataLoader objects for batching and loading data in parallel.
"""

import torch
import torchvision

import requests
import zipfile
from pathlib import Path

from typing import Tuple, List

def download_data() -> None:
    """
    Downloads and extracts a dataset of pizza, steak, and sushi images 
    if not already present in the './data/' directory.

    If the directory doesn't exist, it creates it, downloads the dataset 
    from a specified URL, and extracts the contents of the ZIP file.

    Raises:
        requests.exceptions.RequestException: If the download request fails.
        zipfile.BadZipFile: If the downloaded file is not a valid ZIP file.
    """
    data_path = Path('./data/')
    
    # Checking and creating the data directory
    if data_path.is_dir():
        "Data has been donwloaded previously."
    else:
        data_path.mkdir(parents=True, exist_ok=True)
        
        print('Downloading the data...')
        
        # Getting the zipfile of the data
        zipfile_path = data_path / 'pizza_steak_sushi.zip'
        with open(zipfile_path, 'wb') as f:
            request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi_20_percent.zip')
            f.write(request.content)
        
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)

def get_dataloaders(train_dir: str | Path,
                    test_dir: str | Path,
                    train_transforms: torchvision.transforms,
                    test_transforms: torchvision.transforms,
                    batch_size: int,
                    num_workers: int) -> Tuple[torch.utils.data.DataLoader, 
                                               torch.utils.data.DataLoader, 
                                               List[str]]:
    """
    Loads and returns PyTorch DataLoaders for training and testing datasets, 
    along with class names from the training set.

    Args:
        train_dir (str | Path): Path to the directory containing the training images.
        test_dir (str | Path): Path to the directory containing the testing images.
        train_transforms (torchvision.transforms): Transformations to apply to the training images.
        test_transforms (torchvision.transforms): Transformations to apply to the testing images.
        batch_size (int): Number of samples per batch for the DataLoader.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[str]]:
            - A DataLoader for the training dataset.
            - A DataLoader for the testing dataset.
            - A list of class names (labels) from the training dataset.
    """
    # Get the train and test datasets
    train_data = torchvision.datasets.ImageFolder(root=train_dir,
                                                  transform=train_transforms,
                                                  target_transform=None)
    
    test_data = torchvision.datasets.ImageFolder(root=test_dir,
                                                 transform=test_transforms,
                                                 target_transform=None)
    
    # Getting the class_names of the dataset
    class_names = train_data.classes
    
    # Dataset to Dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   pin_memory=True)
    
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=True)

    return train_dataloader, test_dataloader, class_names