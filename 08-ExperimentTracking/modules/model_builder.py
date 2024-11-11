import torch

from torchinfo import summary

from typing import List

class TinyVGGMiniFoodModel(torch.nn.Module):
    """
    A small CNN model inspired by the Tiny-VGG architecture designed for 
    image classification tasks. This model is optimized for small-scale 
    datasets working on image classification.

    The model consists of two convolutional blocks followed by a fully connected 
    classification layer. It uses convolutional layers with ReLU activations and 
    max-pooling operations to extract features from the input image. The output of the 
    convolutional layers is flattened and passed through a fully connected layer to 
    predict the class probabilities.

    Attributes:
        block_1 (torch.nn.Sequential): The first convolutional block, consisting of 
                                       two convolutional layers with ReLU activations and 
                                       a max-pooling layer.
        block_2 (torch.nn.Sequential): The second convolutional block, similar to the first, 
                                       but with feature map transformations.
        classifier (torch.nn.Sequential): The fully connected layer that flattens the output 
                                          and produces class predictions.

    Args:
        in_channels (int): The number of input channels (e.g., 3 for RGB images).
        out_features (int): The number of output features or classes in the classification task.
        hidden_channels (int): The number of channels in the hidden layers (used in both convolutional blocks).
        image_height (int): The height of the input image.
        image_width (int): The width of the input image.

    Methods:
        forward(x): Defines the forward pass of the model, applying the convolutional blocks 
                    and the classification layer to the input tensor `x`.

    Example:
        model = TinyVGGMiniFoodModel(in_channels=3, out_features=10, hidden_channels=64, image_height=64, image_width=64)
        output = model(input_tensor)
    """
    def __init__(self,
                 in_channels: int,
                 out_features: int,
                 hidden_channels: int,
                 image_height: int,
                 image_width: int) -> None:
        super().__init__()
    
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=hidden_channels * int((image_height/2)/2) * int((image_width/2)/2),
                            out_features=out_features) 
        )
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifer(x)
        return x