import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import scipy.io


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_dim=128, hid_dim=256, out_dim=128):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Linear(in_dim,hid_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim,out_dim),
            # nn.ReLU(),
        )
        # self.Batchnorm = nn.BatchNorm1d(1)

    def forward(self, x):
        out = self.left(x)+x
        # out = torch.unsqueeze(out,dim=1)
        # out = self.Batchnorm(out)
        # out = torch.squeeze(out)
        return out

class En_net(torch.nn.Module):
    def __init__(self,ResBlock):
        super(En_net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(45,256),
            nn.ReLU(),
            nn.Linear(256,1024),
            nn.GELU(),
        )
        self.layer1 = self.make_layer(ResBlock, 1024,2048,1024)
        self.layer2 = self.make_layer(ResBlock, 1024,2048,1024)
        self.layer3 = self.make_layer(ResBlock, 1024,2048,1024)
        self.layer4 = self.make_layer(ResBlock, 1024,2048,1024)

        self.to_deep = nn.Sequential(
            nn.Linear(1024,64),
            nn.Tanh()
        )
    def make_layer(self, block, in_dim=128, hid_dim=256, out_dim=128 ):
        layers = []
        layers.append(block(in_dim,hid_dim,out_dim))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.encoder(x)
        tem = x.detach()
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        x = torch.relu(x)
        x = x + tem
        x = torch.relu(x)
        x = self.to_deep(x)
        return x

class De_net(torch.nn.Module):
    def __init__(self,ResBlock):
        super(De_net, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,1024),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResBlock, 1024,2048,1024)
        self.layer2 = self.make_layer(ResBlock, 1024,2048,1024)
        self.layer3 = self.make_layer(ResBlock, 1024,2048,1024)
        self.layer4 = self.make_layer(ResBlock, 1024,2048,1024)

        self.deep_to = nn.Sequential(
            nn.Linear(1024, 45),
        )
    def make_layer(self, block, in_dim=128, hid_dim=256, out_dim=128 ):
        layers = []
        layers.append(block(in_dim,hid_dim,out_dim))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.decoder(x)
        tem = x.detach()
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        x = torch.relu(x)
        x = x+tem
        x = torch.relu(x)
        x = self.deep_to(x)
        return x

def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    x = torch.randn([256,2])*2
    x2 = torch.randn([256,2])*3+20
    # x3 = torch.randn([256,2])*3+7
    # x4 = torch.randn([256,2])*2+13
    x = torch.cat([x,x2],dim=0)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    # x = x.cpu().numpy()
    encoder = En_net(ResBlock)
    decoder = De_net(ResBlock)
    x = encoder(x)
    x = decoder(x)
    x = x.detach().numpy()
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

if __name__ == '__main__':
    main()