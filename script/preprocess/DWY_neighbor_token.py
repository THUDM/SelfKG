import settings
import pandas as pd
from loader.DWY_Neighbor import NeighborsLoader
from loader.DBP15k import DBP15kLoader
from script.preprocess.get_token import Token
from settings import *
import numpy as np
import torch

class NeighborToken(object):
    def __init__(self, dbpToken, loader):
        self.loader = loader
        self.id_features_dict = dbpToken.id_features_dict

        self.id_adj_tensor_dict = {}

        def get_adj(valid_len):
            adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
            for i in range(0, valid_len):
                adj[i, i] = 1
                adj[0, i] = 1
                adj[i, 0] = 1
            return adj

        def get_token():
            id_neighbors_dict = {}
            loader_id_neighbors_dict_copy = self.loader.id_neighbors_dict


            for entity_id, neighbors_dict in loader_id_neighbors_dict_copy.items():
                id_neighbors_dict[entity_id]=[]
                id_neighbors_dict[entity_id].append(self.id_features_dict[entity_id])
                for rel, neighbor in neighbors_dict.items():
                    for neigh in neighbor:
                        id_neighbors_dict[entity_id].append(self.id_features_dict[neigh])
            for k, v in id_neighbors_dict.items():
                if len(v) < NEIGHBOR_SIZE:
                    self.id_adj_tensor_dict[k] = get_adj(len(v))
                    id_neighbors_dict[k] = v + [[ord(' ')]*TOKEN_LEN] * (NEIGHBOR_SIZE - len(v))
                else:
                    self.id_adj_tensor_dict[k] = get_adj(NEIGHBOR_SIZE)
                    id_neighbors_dict[k] = v[:NEIGHBOR_SIZE]
            return id_neighbors_dict

        self.id_neighbors_dict = get_token()


