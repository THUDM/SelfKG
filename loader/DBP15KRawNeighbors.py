from settings import *
import csv
import pandas as pd
import torch
import pickle

class DBP15KRawNeighbors():
    def __init__(self, language, doc_id):
        self.language = language
        self.doc_id = doc_id
        self.path = join(DATA_DIR, 'DBP15K', self.language)
        self.id_entity = {}
        # self.id_neighbor_loader = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.load()
        self.id_neighbors_loader()
        self.get_center_adj()

    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity = pickle.load(f)


    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']
        # self.id_neighbor_loader = {head: {relation: [neighbor1, neighbor2, ...]}}

        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            # print("int(row['head']): ", int(row['head']))
            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_entity[int(row['tail'])][0]

            if not int(row['head']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['head'])] = [head_str]
            if not tail_str in self.id_neighbors_dict[int(row['head'])]:
                self.id_neighbors_dict[int(row['head'])].append(tail_str)
            
            if not int(row['tail']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['tail'])] = [tail_str]
            if not head_str in self.id_neighbors_dict[int(row['tail'])]:
                self.id_neighbors_dict[int(row['tail'])].append(head_str)
            

            # if not self.id_neighbor_loader.__contains__(head_str):
            #     self.id_neighbor_loader[head_str] = {}
            #     self.id_neighbors_dict[head_str] = [head_str]
            # if not self.id_neighbor_loader[head_str].__contains__(row['relation']):
            #     self.id_neighbor_loader[head_str][row['relation']] = []
            # if not tail_str in self.id_neighbor_loader[head_str][row['relation']]:
            #     self.id_neighbor_loader[head_str][row['relation']].append(tail_str)
            #     self.id_neighbors_dict[head_str].append(tail_str)

            # tail-rel-head, head is a neighbor of tail
            # if not self.id_neighbor_loader.__contains__(tail_str):
            #     self.id_neighbor_loader[tail_str] = {}
            #     self.id_neighbors_dict[tail_str] = [tail_str]
            # if not self.id_neighbor_loader[tail_str].__contains__(row['relation']):
            #     self.id_neighbor_loader[tail_str][row['relation']] = []
            # if not head_str in self.id_neighbor_loader[tail_str][row['relation']]:
            #     self.id_neighbor_loader[tail_str][row['relation']].append(head_str)
            #     self.id_neighbors_dict[head_str].append(head_str)
    
    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):
        for k, v in self.id_neighbors_dict.items():
            if len(v) < NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.id_neighbors_dict[k] = v + [[0]*LaBSE_DIM] * (NEIGHBOR_SIZE - len(v))
            else:
                self.id_adj_tensor_dict[k] = self.get_adj(NEIGHBOR_SIZE)
                self.id_neighbors_dict[k] = v[:NEIGHBOR_SIZE]
        