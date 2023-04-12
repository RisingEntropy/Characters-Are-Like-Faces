import base64
import os
import random
import sys
from io import BytesIO

import h5py
import pygame.image
import torch
import requests
import json

import tqdm
from torchvision import transforms
from fontTools.ttLib import TTFont

from modules import MobileFaceNet

URL = "http://localhost:8088/ai/detect"
pygame.init()
test_count = 5000
words = []
ranges = [(0x4E00, 0x9FCB), (0x3400, 0x4DB5), (0x20000, 0x2A6D6), (0x2A700, 0x2B734), (0x2B740, 0x2B81D)]

ft = pygame.font.Font("fonts/KaiXinSongA.ttf", 112)
ft2 = pygame.font.Font("fonts/KaiXinSong2.1.ttf", 112)
ft_mp = TTFont("fonts/KaiXinSongA.ttf")["cmap"].tables[0].ttFont.getBestCmap()
ft2_mp = TTFont("fonts/KaiXinSong2.1.ttf")["cmap"].tables[0].ttFont.getBestCmap()
for r in ranges:
    for ch in range(r[0], r[1] + 1):
        if ch in ft_mp or ch in ft2_mp:
            words.append(ch)
transform = transforms.ToPILImage()


def img2Base64(img):
    img = pygame.surfarray.array3d(img)
    img = img.transpose((1, 0, 2))
    img = transform(img)
    bio = BytesIO()
    img.save(bio, format='JPEG')
    return base64.b64encode(bio.getvalue())


def postReq(img):
    data = {"img": img2Base64(img)}
    return json.loads(requests.post(URL, data=data).text)


font_names = {}
fonts = []
font_mps = {}
for file in os.listdir("./eval_fonts"):
    fonts.append(pygame.font.Font(os.path.join("./eval_fonts", file), 90))
    font_names[fonts[-1]] = file
    font_info = TTFont(os.path.join("./eval_fonts", file))
    font_mps[fonts[-1]] = font_info["cmap"].tables[0].ttFont.getBestCmap()

for font in fonts:
    top1 = 0
    top_rge = 0
    for iter in tqdm.tqdm(range(test_count), file=sys.stdout):
        word = random.choice(words)
        while word not in font_mps[font]:
            word = random.choice(words)
        word = chr(word)
        img = font.render(word, True, (0, 0, 0), (255, 255, 255))
        img = pygame.transform.scale(img, (112, 112))
        res = postReq(img)
        if res['bestMatch'] == word:
            top1 += 1
            top_rge += 1
        else:
            for cand in res['candidates']:
                if word in cand.keys():
                    top_rge += 1
                    break
    print(f"font: {font_names[font]}, top1:{top1 / test_count:.5f}, in_candidate:{top_rge / test_count:.5f}")
