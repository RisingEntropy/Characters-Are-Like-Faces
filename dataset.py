from torch.utils.data import Dataset
import h5py
import numpy as np


class OCRDataset(Dataset):
    def __init__(self, file_path, transforms):
        self.file = h5py.File(file_path, "r")
        self.imgs = np.asarray(self.file["imgs"])
        self.labels = np.asarray(self.file["labels"])
        self.transforms = transforms

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        if self.transforms is None:
            return self.imgs[item].transpose((2, 0, 1)) / 255, self.labels[item]-0x4E00
        else:
            return self.transforms(self.imgs[item]).transpose((2, 0, 1)) / 255, self.labels[item]-0x4E00
