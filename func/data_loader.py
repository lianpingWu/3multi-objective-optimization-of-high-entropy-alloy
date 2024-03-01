# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, target, flag, use_list, path):
        self.target = target
        self.use_list = use_list
        
        dataset = np.load(path + "samples_" + flag + "_" + target + ".npy")
        samples = dataset[:, 0:22]
        performance = dataset[:, 22]
        
        self.X_data = torch.from_numpy(np.expand_dims(samples, axis=1)).float()
        self.y_data = torch.from_numpy(np.expand_dims(performance, axis=1)).float()
        
    def __len__(self):
        return self.X_data.shape[0]
        
    def __getitem__(self, index):
        return self.X_data[index,:, self.use_list], self.y_data[index,:]


def data_loader(flag, use_list, batch, path):
    train_dataloader = DataLoader(
        dataset=MyDataset(flag, "train", use_list, path),
        batch_size=batch,
        shuffle=True
        )
    
    valid_dataloader = DataLoader(
        dataset=MyDataset(flag, "valid", use_list, path),
        batch_size=1,
        shuffle=True
        )
    
    support_set_path = path + "samples_support_" + flag + ".npy"
    support_set = np.expand_dims(np.load(support_set_path), axis=1)
    support_set = torch.from_numpy(support_set[:, :, use_list]).float()

    return train_dataloader, valid_dataloader, support_set


if __name__ == "__main__":
    use_list = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13] + [17, 18, 19, 20, 21]
    dataloader, support_set = data_loader("YM", use_list, "./dataset/")
        
    print(support_set.shape)
    for coords, labels in dataloader:
        print(coords.size())
        break