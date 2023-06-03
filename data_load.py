import numpy as np
from scipy import io

def split_data(data,label,split_ratio):
    label = label[0,:]
    l = np.shape(label)[0]
    indice = np.array(list(range(l)))
    np.random.shuffle(indice)
    select_indice = indice[:int(l*split_ratio)]
    test = [data[select_indice,:],label[select_indice]]
    data = np.delete(data,select_indice,axis=0)
    label = np.delete(label,select_indice,axis=0)
    train = [data,label]
    return train,test

def load_data(split_ratio):
    # data_dic = scipy.io.loadmat('D:/360安全浏览器下载/资料文件/0-待完成任务/毕设/金海豚PDW数据型/数据处理/kernal_train.mat')
    data_dic = io.loadmat('standard_data.mat')
    data = data_dic['data']
    label = data_dic['label']
    data = np.array(data)
    label = np.array(label) - 1
    data = data[:-2000]
    label = label[:,:-2000]
    return split_data(data,label, split_ratio)

def main():
    load_data(0.25)


if __name__ == '__main__':
    main()

