import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import os
import glob
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.models import mobilenet_v3_small, mobilenet_v2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

device = torch.device("mps")

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        self.backbone = mobilenet_v3_small(pretrained=True)
        for name, para in self.backbone.named_parameters():
            para.requires_grad = False
        classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=50, bias=True),
            nn.Hardswish(),
            nn.Linear(in_features=50, out_features=1, bias=True),
          )
        self.backbone.classifier = classifier
        # summary(self.backbone,input_size=(3,224,224))
    def forward(self, x):
        return self.backbone(x)


def deblur_compare(im1, im2):
    model = DeblurNet()
    model.load_state_dict(torch.load("./deblurnet.pth"))
    model.to(device)
    model.eval()

    tr = transforms.ToTensor()
    image = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB), (224, 224))

    image = tr(image).to(torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        res = model(image).cpu().detach().numpy()[0][0]
    return res


class BlurDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, eval_dataset ="rsblur", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(f"subj_{eval_dataset}.csv")
        videos = self.data.video.unique()
        self.root_dir = f"./crops_{eval_dataset}"
        self.paths = []
        for video in videos:
            lst = os.listdir(os.path.join(self.root_dir, video))
            for name in lst:
                self.paths.append((video, name))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video, name = self.paths[idx]
        image = cv2.imread(os.path.join(self.root_dir, video, name))
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
        tr = transforms.ToTensor()
        image = tr(image)

        name = name.replace(".png", '')

        data_test = self.data[self.data.video == video]
        label = data_test.loc[data_test.method == name].value.values[0]

        if self.transform:
            image = self.transform(image)

        return image.to(torch.float32), torch.from_numpy(np.array([label])).to(torch.float32)

batch_size = 128
validation_ratio = 0.1
random_seed = 10

if __name__ == '__main__':
    DeblurNet()
    net = DeblurNet()
    dataset = BlurDataset()



    train, valid = random_split(dataset, [1 - validation_ratio, validation_ratio])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0)

    lr = 0.001

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    device = torch.device("mps")
    print(device)

    net.to(device)
    min_ble = torch.inf
    for epoch in range(1000):
        running_loss = 0.0
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
        running_loss /= len(train_loader)
        print(running_loss)
        total = 0
        correct = 0
        ble = 0
        for data in tqdm(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            predicted = outputs.data
            ble += torch.abs(predicted - labels).sum()

        if ble < min_ble:
            torch.save(net.state_dict(), "deblurnet2.pth")
        print('[%d epoch] Sum Error of the network on the validation images: %d' %
              (epoch, ble)
             )

    print('Finished Training')
