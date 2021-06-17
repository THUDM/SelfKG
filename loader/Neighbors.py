from settings import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class NeighborsLoader(object):
    def __init__(self, language, doc_id):
        # whether it is in training
        self.language = language
        self.doc_id = doc_id

        self.path = join(DATA_DIR, 'DBP15K', self.language)
        # load id_features
        self.id_neighbors = None
        self.id_neighbors_dict = {}
        self.id_neighbors_loader()

    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['id', 'relation', 'neighbor']
        self.id_neighbors = data
        # self.id_neighbors_dict = {id: {relation: [neighbor1, neighbor2, ...]}}
        for index, row in data.iterrows():
            if not self.id_neighbors_dict.__contains__(row['id']):
                self.id_neighbors_dict[row['id']] = {}
            if not self.id_neighbors_dict[row['id']].__contains__(row['relation']):
                self.id_neighbors_dict[row['id']][row['relation']] = []
            self.id_neighbors_dict[row['id']][row['relation']].append(row['neighbor'])
        # print(self.id_neighbors_dict)
                
            

