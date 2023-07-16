import torch
import torch.nn as nn
import argparse
import json
import sys

sys.path.insert(0, "../")

from utils.model import get_model_dummy, get_model

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to the config file")
args = parser.parse_args()

# Load the config file
config_path = args.config
with open(config_path) as config_file:
    config = json.load(config_file)

# Extract the hyperparameters and device configuration
hyperparameters = config['hyperparameters']
device = config['device']
mode = config['mode']

# Define your loss function
loss_fn = nn.CrossEntropyLoss()

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        # Move the data and target tensors to the device
        data = data.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = loss_fn(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_accuracy += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_accuracy / len(dataloader.dataset)

    return avg_loss, accuracy

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move the data and target tensors to the device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output, target)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_accuracy += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_accuracy / len(dataloader.dataset)

    return avg_loss, accuracy

def training_loop(model, train_dataloader, val_dataloader, optimizer, num_epochs):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train(model, train_dataloader, optimizer)
        val_loss, val_accuracy = evaluate(model, val_dataloader)

        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

        # Check if the current epoch's accuracy is better than the best accuracy so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Save the model checkpoint
            torch.save(model.state_dict(), "best_model.pth")

    print("Training completed!")

def test(model, test_dataloader):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            # Move the data and target tensors to the device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output, target)

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_accuracy += (predicted == target).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_dataloader.dataset)
    accuracy = total_accuracy / len(test_dataloader.dataset)

    return avg_loss, accuracy

# Example usage
if(mode == "dummy"):
    model = get_model_dummy(device)
else:
    model = get_model(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
train_dataloader = get_train_dataloader(hyperparameters['batch_size'])  # Replace with your function to get the train dataloader
val_dataloader = get_val_dataloader(hyperparameters['batch_size'])  # Replace with your function to get the validation dataloader
test_dataloader = get_test_dataloader(hyperparameters['batch_size'])  # Replace with your function to get the test dataloader

# Training loop
training_loop(model, train_dataloader, val_dataloader, optimizer, hyperparameters['num_epochs'])

# Testing
test_loss, test_accuracy = test(model, test_dataloader)
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
