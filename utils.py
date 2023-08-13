from torch.utils.data import SubsetRandomSampler, DataLoader

batch_size = 32

def get_data_loaders(dataset, partition_indices, partition_index):
    train_indices = partition_indices[partition_index]["train"]
    valid_indices = partition_indices[partition_index]["valid"]
    test_indices = partition_indices[partition_index]["test"]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

    return train_loader, valid_loader, test_loader