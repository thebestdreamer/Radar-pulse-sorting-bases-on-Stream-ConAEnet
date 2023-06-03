#采用已训练好的数据和模型架构进行数据处理
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import scipy.io
import sklearn.cluster as cluster
from net import ResBlock,En_net,De_net

def K_means(data,label_ls,num_classes):


    SSE = []  # 存放每次结果的误差平方和
    for k in range(5, 14):
        estimator = cluster.KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(data)
        SSE.append(estimator.inertia_)
    X = range(5, 14)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


    clu = cluster.KMeans(n_clusters=num_classes, random_state=42)
    y_pred = clu.fit_predict(data[:, :])
    y_ls = list(y_pred)

    dic_pred = np.zeros([num_classes, num_classes])
    for i in range(len(label_ls)):
        dic_pred[y_ls[i],label_ls[i]-1] += 1

    dic_pred = swap(dic_pred)
    print(dic_pred)
    acc = np.zeros([dic_pred.shape[0], 1])
    sum_num = 0
    for i in range(dic_pred.shape[0]):
        acc[i,0] = dic_pred[i, i] / int(np.sum(dic_pred[:,i]))
        sum_num += dic_pred[i,i]
        print("第{:d}类正确率:{:.4f}".format(i, acc[i,0]))
    print('平均正确率{:.4f}'.format(acc.mean()))
    print('总体正确率{:.4f}'.format(sum_num/dic_pred.sum()))


    # sklearn自带算法  DBI的值最小是0，值越小，代表聚类效果越好。
    cluster_score_DBI = metrics.davies_bouldin_score(data, y_pred)
    cluster_score_real_DBI = metrics.davies_bouldin_score(data, label_ls[:])
    cluster_score_NMI = metrics.normalized_mutual_info_score(label_ls[:], y_pred)
    cluster_score_F = metrics.f1_score(label_ls[:], y_pred, average='micro')
    print("cluster_score_DBI:", cluster_score_DBI)
    print('cluster_score_real_DBI', cluster_score_real_DBI)
    print("cluster_score_NMI:", cluster_score_NMI)
    print('cluster_score_F:', cluster_score_F)

    return acc.mean()

def swap(x):
    for i in range(x.shape[0]):
        t = np.argmax(x[i,:])
        if x[i,t]>x[t,t]:
            tem = x[:,i].copy()
            x[:,i] = x[:,t]
            x[:,t] = tem
    return x

class MyDataset(Dataset):

    def __init__(self,data,label):
        self.x = data
        self.y = label
        self.len = len(label)

    def __getitem__(self, index):
        return self.x[index,:], self.y[index]

    def __len__(self):
        return self.len

def load_data():
    np.random.seed(42)

    # data_dic = scipy.io.loadmat('D:/360安全浏览器下载/资料文件/0-待完成任务/毕设/金海豚PDW数据型/数据处理/kernal_train.mat')
    data_dic = scipy.io.loadmat('standard_data.mat')
    data = data_dic['data']
    label = data_dic['label']
    label = label[0,:]
    data = np.array(data)
    label = np.array(label)
    data = data[:-2000]
    label = label[:-2000]
    print(data.shape)

    # mean = data.mean(0)
    # std = data.std(0)
    # for i in range(0, data.shape[0]):
    #     data[i, :] = (data[i, :] - mean) / std

    return data,label

def val(encoder,testset,num_classes):
    feature = []
    label_ls = []
    encoder = encoder.to('cpu')
    for data,label in testset:
        data = torch.from_numpy(np.float32(data))
        # data = data.to('cuda')
        # feature = feature + [(item/torch.norm(item,p=2,dim=0)).detach().numpy() for item in encoder(data)]
        feature = feature + [(item).detach().numpy() for item in
                             encoder(data) / torch.unsqueeze(torch.norm(encoder(data), p=2, dim=1), dim=1)]
        label_ls = label_ls + [item.detach().numpy() for item in label[:]]

    feature = np.array(feature)
    label_ls = np.array(label_ls)
    label_ls = label_ls.flatten()

    acc = K_means(feature,label_ls,num_classes)
    # return acc

def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    data, label = load_data()
    label = np.array(label)
    label = np.transpose(label)
    dataset = MyDataset(data, label)

    num_classes = 7

    test_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    Encoder_net = En_net(ResBlock)
    Encoder_net = torch.load('Encoder.pth')
    Encoder_net = Encoder_net.eval()
    # para = torch.load('cenloss_En.pth')
    # Encoder_net.load_state_dict(para)
    val(Encoder_net, test_loader,num_classes)

if __name__ == '__main__':
    main()