import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm
import scipy.io
from net import ResBlock,En_net,De_net
import math
from data_load import load_data
from eval import val
from buffer import Buffer

class MyDataset(Dataset):

    def __init__(self,data,label):
        self.x = data
        self.y = label
        self.len = len(label)

    def __getitem__(self, index):
        return self.x[index,:], self.y[index]

    def __len__(self):
        return self.len

def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def swap(x):
    for i in range(x.shape[0]):
        t = np.argmax(x[i,:])
        if x[i,t]>x[t,t]:
            tem = x[:,i].copy()
            x[:,i] = x[:,t]
            x[:,t] = tem
    return x

class loss_record():
    def __init__(self):
        self.val = []

    def mean(self):
        return sum(self.val)/len(self.val)

    def clear(self):
        self.val = []

    def update(self,x):
        self.val.append(x)

def augment(x):
    #return x1,x2
    #本函数实现输入中间特征的数据增广
    shuffle_index = torch.randperm(x.size()[0])
    aug = x[shuffle_index,:]
    noise = torch.randn(x.size()).to('cuda')*0.05
    x_aug =  x*0.9 + noise + aug*0.1
    x = x + torch.randn(x.size()).to('cuda')*0.05
    return x,x_aug

def make_image(mid_fea,y):
    pca = PCA(2)
    # data_pca = pca.fit_transform(mid_fea[:, :-1])
    # plt.scatter(data_pca[:, 0], data_pca[:, 1], c=y)
    plt.scatter(mid_fea[:, 0], mid_fea[:, 1], c=y)
    plt.show()

def train(train_loader,test_loader,batch_size):
    batch_size = batch_size
    #使特征分散
    alpha = 0.1
    #使特证聚合
    beta = 0.9
    gamma = 1
    ita = 1
    base_learning_rate = 1e-4
    n_epochs = 50
    encoder = En_net(ResBlock)
    decoder = De_net(ResBlock)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_cls = loss_record()
    plt_train_loss = loss_record()
    buf = Buffer()
    train_loss = []
    test_loss = []
    flag = 0
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    optim_en = torch.optim.Adam(encoder.parameters(), lr=base_learning_rate * batch_size / 256, betas=[0.9, 0.99],
                                weight_decay=0.01)
    optim_de = torch.optim.Adam(decoder.parameters(), lr=base_learning_rate * batch_size / 256, betas=[0.9, 0.99],
                                weight_decay=0.01)
    for data_rank,[data, label] in enumerate(train_loader):
        data = torch.from_numpy(np.float32(data))
        data = data.to(device)
        #if buf.memory != []:
            #data = torch.cat((data,buf.memory),dim=0)
        with tqdm(total=n_epochs, desc=f'stream{data_rank}/{len(train_loader)}', postfix=dict,
                  mininterval=0.3) as pbar:
            for epoch in range(n_epochs):
                # 当前 n_batches 个小批次训练数据上的平均损失

                # 当前 n_batches 个小批次训练数据上的平均损失
                loss_cls.clear()
                plt_train_loss.clear()
                # 对梯度进行清零操作，防止错误的梯度累加
                encoder.train()
                decoder.train()
                optim_en.zero_grad()
                optim_de.zero_grad()
                # 前向传播
                data, data_aug = augment(data)
                mid_fea = encoder(data)
                mid_aug_fea = encoder(data_aug)
                # L2
                mid_fea_norm = mid_fea / torch.unsqueeze(torch.norm(mid_fea, p=2, dim=1), dim=1)
                mid_aug_fea_norm = mid_aug_fea / torch.unsqueeze(torch.norm(mid_aug_fea, p=2, dim=1), dim=1)
                # stop_gard
                mid_fea_norm_detach = mid_fea_norm.detach()
                mid_aug_fea_norm_detach = mid_aug_fea_norm.detach()
                re_data = decoder(mid_aug_fea)
                # 交叉相关矩阵
                c = torch.matmul(mid_fea_norm, mid_aug_fea_norm_detach.T) * 0.5
                c = c + torch.matmul(mid_fea_norm_detach, mid_aug_fea_norm.T) * 0.5
                # 蒸馏误差
                if data_rank>0:
                    loss_distillation = -torch.cosine_similarity(encoder(buf.memory),buf.mem_fea,dim=0).mean()
                else:
                    loss_distillation = 0
                # LLLmixup
                shuffle_index = torch.randperm(data.size()[0])
                aug = data[shuffle_index, :]
                loss_LLL = -torch.cosine_similarity(encoder(0.7*data+0.3*aug), 0.7*encoder(data)+0.3*encoder(aug), dim=0).mean()

                # 计算损失
                loss_re = (re_data - data).pow(2).sum()
                loss = (c - torch.eye(data.size()[0]).to(device)).pow(2)
                # 取非对角元素
                loss = loss.diagonal().sum() + alpha * off_diag(loss).sum() / loss.size()[0] + beta * loss_re /loss.size()[0] + gamma*loss_distillation + ita*loss_LLL
                loss_cls.update(loss.cpu().detach().numpy())
                plt_train_loss.update(loss.cpu().detach().numpy())
                # 反向传播并更新模型参数
                loss.backward()
                optim_en.step()
                optim_de.step()
                pbar.set_postfix(**{'train_loss_': loss_cls.mean()})
                pbar.update(1)
                train_loss.append(plt_train_loss.mean())
            #每个batch流出后更新暂存器的记忆和特征
            buf.update_memory(data.detach())
            buf.update_mem_fea(encoder(buf.memory).detach())
            test_loss.append(test(test_loader, encoder, decoder, batch_size, alpha, beta,gamma,ita,buf))
            pbar.set_postfix(**{'test_loss_': test_loss[-1]})
            if data_rank > 3:
                if test_loss[-1] == min(test_loss) :
                    flag += 1
                    torch.save(encoder, 'T_Encoder.pth')

    print('训练完成')
    val(encoder,test_loader,num_classes=7)
    plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss, color='r', linewidth=1.5, linestyle='-', label='train_loss')
    x = np.linspace(0,len(train_loss),len(test_loss))
    plt.plot(x,test_loss, 'b', label='test_loss')
    plt.legend()
    plt.show()

def test(test_loader,encoder,decoder,batch_size,alpha,beta,gamma,ita,buf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss()
    loss_cls = loss_record()
    encoder.eval()
    decoder.eval()

    for [data, label] in test_loader:
        # 当前 n_batches 个小批次训练数据上的平均损失
        loss_cls.clear()
        # 获取输入数据，这里 data 是形如 [inputs, labels] 的列表
        data = torch.from_numpy(np.float32(data))
        data = data.to(device)
        # 前向传播
        data, data_aug = augment(data)
        mid_fea = encoder(data)
        mid_aug_fea = encoder(data_aug)
        # L2
        mid_fea_norm = mid_fea / torch.unsqueeze(torch.norm(mid_fea, p=2, dim=1), dim=1)
        mid_aug_fea_norm = mid_aug_fea / torch.unsqueeze(torch.norm(mid_aug_fea, p=2, dim=1), dim=1)

        mid_fea_norm_detach = mid_fea_norm.detach()
        mid_aug_fea_norm_detach = mid_aug_fea_norm.detach()
        re_data = decoder(mid_aug_fea)
        # 交叉相关矩阵
        c = torch.matmul(mid_fea_norm, mid_aug_fea_norm_detach.T) * 0.5
        c = c + torch.matmul(mid_fea_norm_detach, mid_aug_fea_norm.T) * 0.5
        # 蒸馏误差
        loss_distillation = torch.cosine_similarity(encoder(buf.memory), buf.mem_fea, dim=0).mean()
        # LLLmixup
        shuffle_index = torch.randperm(data.size()[0])
        aug = data[shuffle_index, :]
        loss_LLL = torch.cosine_similarity(encoder(0.7 * data + 0.3 * aug), 0.7 * encoder(data) + 0.3 * encoder(aug),
                                           dim=0).mean()

        # 计算损失
        loss_re = (re_data - data).pow(2).sum()
        loss = (c - torch.eye(data.size()[0]).to(device)).pow(2)
        # 取非对角元素
        loss = loss.diagonal().sum() + alpha * off_diag(loss).sum() / loss.size()[0] + beta * loss_re / loss.size()[0] + gamma * loss_distillation + ita * loss_LLL
        loss_cls.update(loss.cpu().detach().numpy())
    # print(loss_cls.mean())
    # print('测试完成')
    return loss_cls.mean()

def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    split_radio = 0.2
    train_data, test_data = load_data(split_radio)
    batch_size = 1024
    train_set = MyDataset(train_data[0],train_data[1])
    test_set = MyDataset(test_data[0],test_data[1])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    train(train_loader,test_loader,batch_size)
    #0.72
if __name__ == '__main__':
    main()
