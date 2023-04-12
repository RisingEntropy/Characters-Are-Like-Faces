import os

import cv2
import numpy
import pygame
import h5py
import torch
import torchvision.transforms.transforms
from fontTools.ttLib import TTFont

import modules
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class DSet(Dataset):
    def __init__(self, font_file):
        super().__init__()
        ranges = [(0x4E00, 0x9FCB), (0x3400, 0x4DB5), (0x20000, 0x2A6D6), (0x2A700, 0x2B734), (0x2B740, 0x2B81D)]
        self.data = []
        self.ft = pygame.font.Font(font_file, 112)
        self.ft_mp = TTFont(font_file)["cmap"].tables[0].ttFont.getBestCmap()
        for r in ranges:
            for ch in range(r[0], r[1] + 1):
                if ch in self.ft_mp:
                    self.data.append(ch)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        rtext = self.ft.render(chr(self.data[item]), True, (0, 0, 0), (255, 255, 255))
        img = pygame.transform.scale(rtext, (112, 112))
        img = pygame.surfarray.array3d(img)
        return torch.tensor(img.transpose((2, 1, 0))/255), self.data[item]  # CxHxW


pygame.init()
# ranges = [(0x4E00, 0x4F00)]
net = modules.MobileFaceNet(embedding_size=64).cuda()
net.load_state_dict(torch.load("saved_models/ocrnet_argued_19.pt"))
database = h5py.File("char_database.h5", "w")
net.eval()

dataloaders=[]
word_count = {}
word_feature_vectors = {}
for file in os.listdir("./eval_fonts"):
    dataloaders.append(DataLoader(dataset=DSet(os.path.join("./eval_fonts", file)), batch_size=64, shuffle=False))

for dataloader in dataloaders:
    for it, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = batch[0].float().cuda()
        label = batch[1]
        result = net(imgs)
        for b in range(result.shape[0]):
            if str(label[b].item()) in word_feature_vectors.keys():
                word_feature_vectors[str(label[b].item())] += result[b].detach().cpu().numpy()
                word_count[str(label[b].item())] += 1
            else:
                word_feature_vectors[str(label[b].item())] = result[b].detach().cpu().numpy()
                word_count[str(label[b].item())] = 1

for char, vec in word_feature_vectors.items():
    database.create_dataset(name=char, data=vec/word_count[char])
database.close()
