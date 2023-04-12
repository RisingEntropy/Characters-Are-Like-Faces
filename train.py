import os.path
import sys

import cv2
import numpy
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import modules
import character_argumentation
from dataset import OCRDataset
from tqdm import tqdm
from torchvision import transforms
 
# ---hyper-parameters
lr = 0.001
ocr_net = modules.MobileFaceNet(embedding_size=64).cuda()
arc_net = modules.Arcface(64, 500).cuda()
optim = Adam(
    [{'params': ocr_net.parameters(), 'weight_decay': 5e-4},
     {'params': arc_net.parameters(), 'weight_decay': 5e-4}], lr=lr)
batch_size = 64
total_epoch = 100
save_step = 10
criterion = torch.nn.CrossEntropyLoss()
# --dataset
trs = [character_argumentation.RandomLines(p=0.2)]
# trs = [word_transform.RandomLines(p=0.2), word_transform.RandomRiceGrid(p=0.5)]
train_set = OCRDataset("./train_data.h5", transforms=transforms.Compose(trs))
# test_set = OCRDataset("./train_data.h5", transforms=transforms.Compose(trs))
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# --assistant function
def save(epoch):
    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
    torch.save(ocr_net.state_dict(), f"./saved_models/ocrnet_argued_{epoch}.pt")
    torch.save(arc_net.state_dict(), f"./saved_models/arcnet_argued_{epoch}.pt")


def load(epoch):
    ocr_net.load_state_dict(torch.load(f"./saved_models/ocrnet_argued_{epoch}.pt"))
    arc_net.load_state_dict(torch.load(f"./saved_models/arcnet_argued_{epoch}.pt"))

# load(49)
# --train
for epoch in range(total_epoch):

    train_loss = 0
    train_acc = 0
    test_acc = 0
    train_cnt = 0
    test_loss = 0
    test_cnt = 0
    train_saps = 0
    test_saps = 0
    print(f"epoch {epoch} begins")
    for iter, batch_data in tqdm(enumerate(train_loader), file=sys.stdout, total=len(train_loader)):
        img, label = batch_data[0], batch_data[1]
        # cv2.imshow("img", numpy.uint8(img[0].permute((1, 2, 0)).numpy()*255))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        img = img.float().cuda()
        label = label.long().cuda()
        out1 = ocr_net(img)
        out = arc_net(out1, label)
        loss = criterion(out, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_saps += label.numel()
        train_loss += loss.item()
        train_cnt += 1
        train_acc += (out.argmax(dim=1) == label).float().sum().item()

    print(f"train avg loss:{train_loss / train_cnt}, train avg acc:{train_acc / train_saps}")
    #
    # for iter, batch_data in tqdm(enumerate(test_loader), file=sys.stdout):
    #     img, label = batch_data
    #     img = img.float().cuda()
    #     label = label.long().cuda()
    #     out1 = ocr_net(img)
    #     out = arc_net(out1, label)
    #     loss = criterion(out, label)
    #     test_saps += label.numel()
    #     test_loss += loss.item()
    #     test_cnt += 1
    #     test_acc += (out.argmax(dim=1) == label).float().sum().item()
    #
    # print(f"test avg loss:{test_loss / test_cnt}, test avg acc:{test_acc / test_saps}")
    if ((epoch + 1) % save_step == 0):
        save(epoch)
    print(f"epoch {epoch} finished!")
os.system("shutdown -s -t 0")
