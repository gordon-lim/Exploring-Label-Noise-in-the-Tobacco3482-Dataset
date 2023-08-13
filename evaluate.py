import time
import json
import argparse
from art import *
from utils import *
from train import *
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Evaluate on Tobacco3482')
parser.add_argument('-model', 
                    default = 'vgg16', 
                    choices=['vgg16', 'resnet50', 'googlenet'], 
                    help='The choice CNN to evaluate. The possible values are "vgg16" (default), "resnet50" and googlenet".')
parser.add_argument('-dataset', 
                    default='tobacco3482', 
                    choices=['tobacco3482', 'cleantobacco'],
                    help='The possible values are "tobacco3482" (default) and "cleantobacco" with label errors removed.')
parser.add_argument('--reproduce', action='store_true', help='Include this flag to reproduce benchmark results from paper.')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# TODO: Download Tobacco3482 to directory
data_dir = 'Tobacco3482-jpg'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# See make-partitions.py
partition_file = open(f'{args.dataset}-partitions.json')
partition_indices = json.load(partition_file)

results = []
OUTPUT_PATH = f'Results/{args.dataset}-{args.model}-results.json'

num_partitions = 5
for partition_num in range(num_partitions):
    
    print(text2art(f'Partition {partition_num}'))

    # Load the DataLoader objects for each split
    train_loader, val_loader, test_loader = get_data_loaders(dataset, partition_indices, partition_num)
    
    # Init model
    if(args.model == "vgg16"):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Freeze layers for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10, bias=True)
    elif(args.model == "googlenet"):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights=models.GoogLeNet_Weights.DEFAULT)
        # Freeze layers for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(in_features=1024, out_features=10, bias=True)
    elif(args.model == "resnet50"):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=models.ResNet50_Weights.DEFAULT)
        # Freeze layers for transfer learning
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)

    if not args.reproduce:
        LOAD_PATH = SAVE_PATH = f'Weights/{args.dataset}-{args.model}-p{partition_num}.pth'
        train_model(model, train_loader, val_loader, SAVE_PATH)
    else:
        LOAD_PATH = f'Weights/{args.dataset}-{args.model}-p{partition_num}-reproduce.pth'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model.load_state_dict(torch.load(LOAD_PATH))
    model = model.to(device)
    print("Model", LOAD_PATH, 'loaded.')
    test_loss = 0.0
    test_acc = 0.0
    for inputs, labels in tqdm(test_loader):
        # move tensors to GPU if CUDA is available
        inputs, labels = inputs.to(device), labels.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs)
        _, preds = torch.max(output.data, 1)
        test_acc += torch.sum(preds == labels.data).item()
        del inputs, labels, output, preds
        torch.cuda.empty_cache()
    test_acc = test_acc / len(test_loader.sampler)
    print("Test Accuracy:", test_acc)
    
    results.append({
        "partition": partition_num,
        "test_accuracy": test_acc
    })

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=4)