import torch
from torch.utils.data import Dataset
import numpy as np


class GPTDataset(Dataset):

    def __init__(self, path, context_length):

        self.tokens = np.fromfile(path, dtype = np.uint16)
        self.context_length = context_length
    
    def __len__(self):

        return len(self.tokens) - self.context_length

    def __getitem__(self, idx):
        end = idx + self.context_length
        if(idx < 0 or end >= len(self.tokens)):
            raise IndexError("Index out of bounds for GPTDataset.")
        x = self.tokens[idx:end]
        y = self.tokens[idx+1:end+1]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


