import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import *
from loader.DBP15k import DBP15kLoader
from script.preprocess.get_token import Token


class Mydataset(Dataset):
    def __init__(self, id_features_dict, adj_tensor_dict,is_neighbor=True):
        super(Mydataset, self).__init__()
        self.num = len(id_features_dict)  # number of samples

        self.x_train = []
        self.x_train_adj = None
        self.y_train = []

        for k in id_features_dict:
            if is_neighbor:
                if self.x_train_adj==None:
                    self.x_train_adj = adj_tensor_dict[k].unsqueeze(0)
                else:
                    self.x_train_adj = torch.cat((self.x_train_adj, adj_tensor_dict[k].unsqueeze(0)), dim=0)
            self.x_train.append(id_features_dict[k])
            self.y_train.append([k])

        # transfer to tensor
        # if type(self.x_train[0]) is list:
        # print("self.x_train_adj.shape: ", self.x_train_adj.shape)
        self.x_train = torch.Tensor(self.x_train).long()
        if is_neighbor:
            self.x_train = torch.cat((self.x_train, self.x_train_adj), dim=2)
        # print("self.x_train.shape: ", self.x_train.shape)
        self.y_train = torch.Tensor(self.y_train).long()

    # indexing
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.num


