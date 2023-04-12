import random

from torchvision.transforms import transforms
import cv2
from random import randint


class RandomLines:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        W = img.shape[0]
        H = img.shape[1]
        if randint(0, 10) < self.p * 10:
            cv2.line(img, (randint(0, W-1), randint(0, H-1)), (randint(0, W-1), randint(0, H-1)),
                            (randint(0, 255), randint(0, 255), randint(0, 255)), randint(1, 5))
            return img
        else:
            return img

    def __repr__(self) -> str:
        return "RandomLines"

class RandomRiceGrid:  # 随机米字格
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        W = img.shape[0]
        H = img.shape[1]
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        if randint(0, 10) < self.p * 10:
            thik = randint(1, 5)
            cv2.line(img, (W//2, 0), (W//2, H-1), color, thik)
            cv2.line(img, (0, H//2), (W-1, H//2), color, thik)
            return img
        else:
            return img

    def __repr__(self) -> str:
        return "Random Rice Grid"