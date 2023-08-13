import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

# Download Tobacco3482 to directory
data_path = 'Tobacco3482-jpg'

# define the transformations to be applied to the images
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageFolderWithFilename(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, filename) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        filename = os.path.basename(path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, filename

# create the ImageFolder dataset from the root folder and the transformations
dataset = ImageFolderWithFilename(root=data_path, transform=transforms)

labels = ["ADVE", "Email", "Form", "Letter", "Memo", "News", "Note", "Report", "Resume", "Scientific"]
id2label = {v: k for v, k in enumerate(labels)}

skf = StratifiedKFold(n_splits=5, shuffle=True)

predictions = []
# Iterate over the folds
for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.imgs, dataset.targets)):

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Freeze layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10, bias=True)
    output_path = "tobacco3482-vgg16-cleanlab.json"

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    num_epochs = 28

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - (epoch / num_epochs))**0.5)

    criterion = nn.CrossEntropyLoss()

    print(f"Fold {fold+1}:")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=32, sampler=train_subsampler, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
                      dataset,
                      sampler=test_subsampler, num_workers=8)

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        ###################
        # train the model #
        ###################
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for inputs, labels, _ in tqdm(train_loader):
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
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
            train_acc += torch.sum(preds == labels.data)

            del inputs, labels, output, preds
            torch.cuda.empty_cache()
        print("Training Loss:", train_loss / len(train_subsampler))
        print("Training Accuracy:", train_acc / len(train_subsampler))

    ######################
    # validate the model #
    ######################
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    for inputs, labels, filename in tqdm(val_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs)
        # CleanLab
        logits = output.data
        predicted_class_idx = torch.argmax(logits, 1).item()
        predicted_class_confidence = torch.nn.Softmax(1)(logits).max().item()
        predictions.append({
        'filename': filename,
        'true_label': id2label[labels.item()],
        'predicted_label': id2label[predicted_class_idx],
        'confidence': predicted_class_confidence,
        'logits': logits.squeeze().tolist()
        })
        # Statistics
        _, preds = torch.max(output.data, 1)
        # calculate the batch loss
        loss = criterion(output, labels)
        # update average validation loss
        val_loss += loss.item()*inputs.size(0)
        val_acc += torch.sum(preds == labels.data)

        del inputs, labels, output, preds
        torch.cuda.empty_cache()
    print("Validation Loss:", val_loss / len(test_subsampler))
    print("Validation Accuracy:", val_acc / len(test_subsampler))

with open(output_path, 'w') as f:
    json.dump(predictions, f, indent=4)