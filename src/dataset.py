import numpy as np
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        img = np.array(img)
        img = self.transform(image=img)["image"]

        return img, int(target)
