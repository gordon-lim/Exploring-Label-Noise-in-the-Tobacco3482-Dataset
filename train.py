import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, SAVE_PATH):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move tensors to GPU if CUDA is available
    model.to(device)
    
    num_epochs = 60

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - (epoch / num_epochs))**0.5)

    criterion = nn.CrossEntropyLoss()

    patience = 5 # Number of epochs to wait before stopping if validation loss doesn't improve
    min_delta = 0.01 # Minimum change in validation loss to count as improvement
    best_val_loss = float('inf') # Best validation loss seen so far
    best_epoch = 0 # Best model so far
    wait = 0 # Number of epochs to wait since last improvement

    for epoch in range(num_epochs):

        t0 = time.time()

        print("Epoch:", epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for inputs, labels in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            inputs, labels = inputs.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            _, preds = torch.max(output.data, 1)
            # calculate the batch loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
            train_acc += torch.sum(preds == labels.data).item()

            del inputs, labels, output, preds
            torch.cuda.empty_cache()
        print("Training Loss:", train_loss / len(train_loader.sampler))
        print("Training Accuracy:", train_acc / len(train_loader.sampler))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for inputs, labels in tqdm(val_loader):
            # move tensors to GPU if CUDA is available
            inputs, labels = inputs.to(device), labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            _, preds = torch.max(output.data, 1)
            # calculate the batch loss
            loss = criterion(output, labels)
            # update average validation loss 
            val_loss += loss.item()*inputs.size(0)
            val_acc += torch.sum(preds == labels.data).item()

            del inputs, labels, output, preds
            torch.cuda.empty_cache()
        print("Validation Loss:", val_loss / len(val_loader.sampler))
        print("Validation Accuracy:", val_acc / len(val_loader.sampler))

        scheduler.step() 

        time_taken = time.time() - t0
        print(f'Time taken for epoch {epoch}: {time_taken} seconds')

        # Check if the validation loss has improved
        delta = best_val_loss - val_loss
        if delta > min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            # Save the model weights
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break