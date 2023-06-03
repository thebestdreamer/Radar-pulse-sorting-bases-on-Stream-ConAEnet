from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

class Buffer():
    def __init__(self):
        self.center = []
        self.memory = []
        self.max_mem_size = 256
        self.mem_fea = []
        # 中心梯度更新
        self.step_alpha  = 0.1
        # 增广相加比例
        self.aug_alpha = 0.05

    def update_memory(self,data):
        #难点 如何用保留数据最大程度代表信息
        if self.memory == []:
            index = torch.randperm(data.size()[0])
            self.memory = data[index[:self.max_mem_size],:]
            self.memory = mass_rank(self.memory, data)
        else:
            self.memory = mass_rank(self.memory,data)
        return self.memory

    def update_mem_fea(self,new_fea):
        self.mem_fea = new_fea

    def update_center(self,data):
        #根据输入数据进行聚类，取聚类中心对既有类中心进行动量更新，返回输入聚类中心
        if self.center == []:
            clu = cluster.KMeans(n_clusters=7, random_state=42)
            data = data.to('cpu')
            clu.fit(data[:, :])
            self.center = torch.from_numpy(clu.cluster_centers_).float()
        else:
            clu = cluster.KMeans(n_clusters=7, random_state=42)
            data = data.to('cpu')
            clu.fit(data[:int(self.max_mem_size*1.2), :])
            new_center = torch.from_numpy(clu.cluster_centers_).float()
            new_center = new_center/torch.linalg.norm(new_center,axis=1).view([-1,1])
            dist = torch.matmul(self.center, new_center.transpose(0,1))
            mem_ind,new_ind = linear_sum_assignment(-dist)
            self.center[mem_ind] = (1-self.step_alpha)*self.center[mem_ind] + self.step_alpha*new_center[new_ind]

    # def memory_aug(self,data):
    #     index = torch.randint(0,self.memory.size()[0],data.size()[0])
    #     rand_select = torch.select(self.memory,dim=0,index=index)
    #     aug_data = (1-self.aug_alpha)*data + self.aug_alpha*rand_select
    #     return aug_data

def rand_select(data,rand_split):
    indice = torch.randperm(data.size()[0])
    return data[indice[:int(rand_split*data.size()[0])]],data[indice[int(rand_split*data.size()[0]):]]


def mass_rank(mem,data):
    rand_split = 0.2
    select_mem, stay_mem = rand_select(mem,rand_split)
    select_data,_ = rand_select(data,rand_split)
    mass_in_data = torch.zeros((select_data.size()[0],1))
    mass_in_mem = torch.zeros((select_mem.size()[0],1))
    for i in range(select_data.size()[0]):
        ref_mem,_ = rand_select(mem,rand_split)
        mass_in_data[i] = -torch.matmul(select_data[i,:],ref_mem[:,:].transpose(0,1)).sum()
    for i in range(select_mem.size()[0]):
        ref_mem,_ = rand_select(mem,rand_split)
        mass_in_mem[i] = -torch.matmul(select_data[i,:],ref_mem[:,:].transpose(0,1)).sum()
    mass_index = torch.cat((mass_in_data,mass_in_mem),dim=0)
    index = torch.argsort(mass_index,dim=0,descending=True)
    select_ = torch.cat((select_data,select_mem),dim=0)
    return torch.cat((torch.squeeze(select_[index[:int(rand_split*mem.size()[0])]]),stay_mem),dim=0)

def main():
    # buffer = Buffer()
    # buffer.center = np.random.randn(6,2)
    # buffer.center = torch.from_numpy(buffer.center)
    # buffer.center = buffer.center/torch.linalg.norm(buffer.center,axis=1).reshape([6,1])
    # # new_center = np.random.randn(6,2)
    # buffer.alpha = 0.1
    # for _ in range(30):
    #     data = torch.randn([60,2])
    #     data = data / torch.linalg.norm(data, axis=1).reshape([60, 1])
    #     new_center = buffer.update_center(data)
    #     buffer.update_center(new_center)
    #     buffer.center = buffer.center / torch.linalg.norm(buffer.center, axis=1).reshape([6, 1])
    #     plt.scatter(buffer.center[:,0], buffer.center[:,1], c='r')
    #     plt.scatter(new_center[:, 0], new_center[:, 1], c='b')
    #     plt.show()
    #     pass


    mem = torch.randn([128,2])
    mem = mem / torch.linalg.norm(mem, axis=1).view([-1,1])
    plt.scatter(mem[:, 0], mem[:, 1], c='g',s=10)
    for _ in range(30):
        center = torch.randn([128, 2])
        center = center / torch.linalg.norm(center, axis=1).view([-1, 1])
        mem = mass_rank(mem,center)
        plt.scatter(mem[:,0],mem[:,1],c = 'r',s = 8)
        plt.show()



if __name__ == '__main__':
    main()