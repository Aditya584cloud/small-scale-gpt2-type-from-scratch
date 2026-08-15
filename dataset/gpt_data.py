import numpy as np
import torch

class GPTData:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def get_batch(self, batch_size, context_length, device):
        starts = torch.randint(0, len(self.data) - context_length - 1, (batch_size,))

        x = torch.stack([
            torch.from_numpy(self.data[i:i + context_length].astype(np.int64)) for i in starts])

        y = torch.stack([torch.from_numpy(self.data[i + 1:i + context_length + 1].astype(np.int64)) for i in starts])

        return x.to(device), y.to(device)