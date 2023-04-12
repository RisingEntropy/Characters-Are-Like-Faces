import h5py
import cv2
import numpy
import torch
import tqdm
from PIL.Image import Resampling, Transpose
from torchvision import transforms
import numpy as np
from PIL import Image
from modules import MobileFaceNet

database = h5py.File("./char_database.h5")
net = MobileFaceNet(embedding_size=64)
net.load_state_dict(torch.load("./saved_models/ocrnet_argued_19.pt"))
net.eval()
img = cv2.imread("test_pic.jpg")
img = cv2.resize(img, (112, 112))
# img = img.rotate(-90).transpose(Transpose.FLIP_LEFT_RIGHT)
img = np.array(img)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
img = torch.tensor(img).permute(2, 0, 1).unsqueeze(dim=0)
embeding = net(img.float()/255).squeeze(dim=0)
bst_error = 1e9
bst_char = None
possible_res = []
for char in tqdm.tqdm(database.keys()):
    err = torch.linalg.norm(embeding - torch.tensor(database[char]))
    if err < bst_error:
        bst_error = err
        bst_char = char
    if err < 0.9:
        possible_res.append(char)

for char in tqdm.tqdm(database.keys()):
    err = torch.linalg.norm(embeding - torch.tensor(database[char]))
    if err < bst_error:
        bst_error = err
        bst_char = char
    if err < 0.9:
        possible_res.append(char)

print(bst_char)
print(bst_error)
print(possible_res)
