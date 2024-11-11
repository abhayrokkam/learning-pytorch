import torch

import torchmetrics

from tqdm.auto import tqdm
from typing import Dict

def train_epoch(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_function: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Dict:
    """
    Trains the model for one epoch.

    Performs forward and backward passes, computes the loss, 
    and updates the model parameters using the optimizer.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The training data.
        loss_function (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        device (torch.device): The device (CPU or GPU) for training.

    Returns:
        Dict: A dictionary containing the average training loss for the epoch.
    """
    # Move model to device
    model = model.to(device)
    
    # Set model to train mode
    model = model.train()
    
    # Variable to average loss per epoch
    train_loss = 0
    
    for X, y in dataloader:
        # Move data to device
        X, y = X.to(device), y.to(device)
        
        # Forward pass -> loss -> zero grad -> back prop -> gradient descent
        y_preds = model(X)
        loss = loss_function(y_preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Adding the calculated loss
        train_loss += loss
    
    train_loss /= len(dataloader)
    return {"train_loss": train_loss.item()}

def test_epoch(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               accuracy_function: torchmetrics.Accuracy,
               device: torch.device) -> Dict:
    """
    Evaluates the model on the test dataset for one epoch.

    The function computes the average loss and accuracy by performing a forward pass 
    through the model on the test data, calculating the loss, and evaluating the accuracy.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
        loss_function (torch.nn.Module): The loss function used to compute the test loss.
        accuracy_function (torchmetrics.Accuracy): The function to compute accuracy.
        device (torch.device): The device (CPU or GPU) on which to run the model.

    Returns:
        Dict: A dictionary with the average test loss ('test_loss') and 
              test accuracy ('test_accuracy') for the epoch.  
    """
    # Move model to the device
    model = model.to(device)
    
    # Set model to eval mode
    model = model.eval()
    
    # Variables to track loss and accuracy
    test_loss = 0
    test_accuracy = 0
    
    for X, y in dataloader:
        # Data to device
        X = X.to(device)
        y = y.to(device)
        
        # Accuracy function to device
        accuracy_function = accuracy_function.to(device)
        
        # Forward pass, loss and accuracy
        y_preds = model(X)
        test_loss += loss_function(y_preds, y)
        test_accuracy += accuracy_function(torch.argmax(y_preds, dim = 1).squeeze(), y)
    
    # Average
    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)
    return {'test_loss': test_loss.item(),
            'test_accuracy': test_accuracy.item()}

def train(epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          accuracy_function: torchmetrics.Accuracy,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict:
    """
    Trains and evaluates the model for a specified number of epochs.

    For each epoch, the model is trained on the training data, and its performance
    is evaluated on the test data. The function tracks training loss, test loss, 
    and test accuracy, and prints the results for each epoch.

    Args:
        epochs (int): The number of epochs to train the model.
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
        loss_function (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used to update model weights.
        accuracy_function (torchmetrics.Accuracy): The function to compute accuracy on the test set.
        device (torch.device): The device (CPU or GPU) on which the model is trained and evaluated.

    Returns:
        Dict: A dictionary containing lists of training loss, test loss, and test accuracy 
              over all epochs. Keys are 'train_loss', 'test_loss', and 'test_accuracy'.
    """
    results = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    # Run through every epoch
    for epoch in tqdm(range(epochs)):
        print(f"\nEPOCH: {epoch} ----------------------------------------------- \n")
        train_dict = train_epoch(model = model,
                                    dataloader=train_dataloader,
                                    loss_function=loss_function,
                                    optimizer=optimizer,
                                    device=device)
        test_dict = test_epoch(model=model,
                                  dataloader=test_dataloader,
                                  loss_function=loss_function,
                                  accuracy_function=accuracy_function,
                                  device=device)
    
        print(f"Epoch: {epoch}  |  Loss: {train_dict['train_loss']:.2f}  |  Test Loss: {test_dict['test_loss']:.2f}  |  Test Accuracy: {test_dict['test_accuracy']:.2f}")
        
        results['train_loss'].append(train_dict['train_loss'])
        results['test_loss'].append(test_dict['test_loss'])
        results['test_accuracy'].append(test_dict['test_accuracy'])
        
        # Updating the values to tensorboard
        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={'train_loss': train_dict['train_loss'],
                                            'test_loss': test_dict['test_loss']},
                           global_step = epoch)
        writer.close()
    
    return results
    