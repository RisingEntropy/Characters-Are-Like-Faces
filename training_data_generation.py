import os
import random

import h5py
import numpy
import pygame
import cv2
import PIL
import tqdm
from fontTools.ttLib import TTFont

fonts = []
for file in os.listdir("./fonts"):
    fonts.append(os.path.join("./fonts", file))
size_range = (15, 112)
pygame.init()
char_count = 500
img_size = (112, 112)
# img_size = (64, 64)
font_mp = {}
# feal_fnt = []
for font in fonts:
    font_info = TTFont(font)
    mp = font_info["cmap"].tables[0].ttFont.getBestCmap()
    font_mp[font] = mp
    # real_fnt.a


def generate_data(name="train"):
    file = h5py.File(name + "_data.h5", 'w')
    images = []
    labels = []
    for char in tqdm.tqdm(range(0x4E00, 0x4E00 + char_count)):
        for sze in range(*size_range, 5):
            for font in fonts:
                if char not in font_mp[font]:
                    continue

                fnt = pygame.font.Font(font, sze)
                surface = pygame.Surface((112, 112))
                surface.fill((255, 255, 255))
                thk = random.randint(1, 5)
                col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pygame.draw.line(surface, col, (112 // 2, 0), (112 // 2, 112), width=thk)
                pygame.draw.line(surface, col, (0, 112 // 2), (112, 112 // 2), width=thk)

                img = fnt.render(chr(char), True, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                img = pygame.transform.scale(img, img_size)

                surface.blit(img, img.get_rect())
                surface = pygame.surfarray.array3d(surface)  # HxWxC
                surface = surface.transpose((1, 0, 2))

                if random.randint(0, 10) < 5:
                    # cv2.imshow("asd", surface)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    images.append(numpy.expand_dims(surface, axis=0))
                else:
                    img = fnt.render(chr(char), True, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                                     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                    img = pygame.transform.scale(img, img_size)
                    img = pygame.surfarray.array3d(img)  # HxWxC
                    img = img.transpose((1, 0, 2))
                    # cv2.imshow("asd", img)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    images.append(numpy.expand_dims(img, axis=0))
                labels.append(char)

    file.create_dataset("imgs", data=numpy.concatenate(images, axis=0))
    file.create_dataset("labels", data=labels)
    file.close()


generate_data("train")
