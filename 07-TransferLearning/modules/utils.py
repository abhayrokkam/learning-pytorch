import torch

from pathlib import Path

import matplotlib.pyplot as plt

from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """
    Plots the training and test loss curves over epochs.

    This function takes the results dictionary containing training and test losses, 
    and plots them on the same graph to visualize the model's performance across epochs.

    Args:
        results (Dict[str, List[float]]): A dictionary with keys 'train_loss' and 'test_loss',
                                          each containing a list of loss values for each epoch.

    Returns:
        None: The function displays a plot of the loss curves but does not return anything.
    """
    # Get loss lists
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    
    epochs = range(len(results['train_loss']))
    
    plt.figure(figsize = (6,4))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves the model's state_dict to a specified directory.

    This function creates the target directory (if it doesn't exist), checks that the 
    model name ends with '.pth' or '.pt', and saves the model's state_dict to the 
    specified path.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_dir (str): The directory where the model will be saved.
        model_name (str): The name of the model file (should end with '.pth' or '.pt').

    Raises:
        AssertionError: If the `model_name` does not end with '.pth' or '.pt'.
    
    Returns:
        None: The function saves the model and does not return anything.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "The argument 'model_name' should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)