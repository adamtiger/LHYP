from torch.utils.data import Dataset
import torch


class LhypDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]['img']

        if self.transform:
            sample = self.transform(sample)

        return (sample, torch.tensor(self.data[idx]['label']))
