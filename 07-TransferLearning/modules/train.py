import os

import torch
import torchvision
from torchvision import transforms

import torchmetrics

from pathlib import Path

import data_setup, model_builder, engine, utils

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = int(os.cpu_count() / 2)
LEARNING_RATE = 0.0001

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

data_path = Path('./data/')
train_dir = data_path / 'train'
test_dir = data_path / 'test'

# Device agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Downloading the data
data_setup.download_data()

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize(size = (IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size = (IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])

# Get the dataloaders
train_dataloader, test_dataloader, class_names = data_setup.get_dataloaders(train_dir=train_dir,
                                                                            test_dir=test_dir,
                                                                            train_transforms=train_transforms,
                                                                            test_transforms=test_transforms,
                                                                            batch_size=BATCH_SIZE,
                                                                            num_workers=NUM_WORKERS)

# Instantiating the model
model = model_builder.TinyVGGMiniFoodModel(in_channels=3,
                                           out_features=len(class_names),
                                           hidden_channels=16,
                                           image_height=IMAGE_HEIGHT,
                                           image_width=IMAGE_WIDTH)

# Loss function, optimizer, accuracy function
loss_function = torch.nn.CrossEntropyLoss()
accuracy_function = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names))
optimizer = torch.optim.Adam(params = model.parameters(),
                             lr = LEARNING_RATE)

# Training the model
engine.train(epochs=EPOCHS,
             model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_function=loss_function,
             optimizer=optimizer,
             accuracy_function=accuracy_function,
             device=device)

# Saving the model
utils.save_model(model=model,
                 target_dir='./models/',
                 model_name='50E_32BS_1e-3LR.pth')