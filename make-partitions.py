import os
import json
import random
import argparse
import pandas as pd
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='Make Tobacco3482 partitions')
parser.add_argument('-dataset', 
                    default='tobacco3482', 
                    choices=['tobacco3482', 'cleantobacco'],
                    help='The possible values are "tobacco3482" (default) and "cleantobacco" with label errors removed.')
args = parser.parse_args()

# TODO: Download Tobacco3482 to directory
data_dir = 'Tobacco3482-jpg'
dataset = datasets.ImageFolder(root=data_dir)

df = pd.read_csv('tobacco3482-errors.csv')
mislabeled = df[df['Mislabeled'] == 1]['filename']

num_partitions = 5
partition_indices = []
for i in range(num_partitions):
    indices = {k: [] for k in range(10)}
    for j, (image_path, class_index) in enumerate(dataset.imgs):
        if args.dataset == 'cleantobacco' and os.path.basename(image_path) not in list(mislabeled):
          indices[class_index].append(j)
    train_indices = []
    valid_indices = []
    for k in range(10):
        random.shuffle(indices[k])
        train_indices += indices[k][:80]
        valid_indices += indices[k][80:100]
    all_indices = set(value for sublist in indices.values() for value in sublist)
    test_indices = list(set(all_indices) - set(train_indices) - set(valid_indices))
    partition_indices.append({
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices
    })

with open(f'{dataset.args}-partitions', 'w') as f:
    json.dump(partition_indices, f, indent=4)