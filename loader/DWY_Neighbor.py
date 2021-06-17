from settings import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class NeighborsLoader(object):
    def __init__(self, language, doc_id):
        # whether it is in training
        self.language = language
        self.doc_id = doc_id

        self.path = join(DATA_DIR, 'DWY100K', self.language)
        # load id_features
        self.id_neighbors = None
        self.id_neighbors_dict = {}
        self.id_neighbors_loader()

    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']
        self.id_neighbors = data
        # self.id_neighbors_dict = {head: {relation: [neighbor1, neighbor2, ...]}}

        for index, row in data.iterrows():

            # head-rel-tail, tail is a neighbor of head
            if not self.id_neighbors_dict.__contains__(row['head']):
                self.id_neighbors_dict[row['head']] = {}
            if not self.id_neighbors_dict[row['head']].__contains__(row['relation']):
                self.id_neighbors_dict[row['head']][row['relation']] = []
            if not row['tail'] in self.id_neighbors_dict[row['head']][row['relation']]:
                self.id_neighbors_dict[row['head']][row['relation']].append(row['tail'])

            # tail-rel-head, head is a neighbor of tail
            if not self.id_neighbors_dict.__contains__(row['tail']):
                self.id_neighbors_dict[row['tail']] = {}
            if not self.id_neighbors_dict[row['tail']].__contains__(row['relation']):
                self.id_neighbors_dict[row['tail']][row['relation']] = []
            if not row['head'] in self.id_neighbors_dict[row['tail']][row['relation']]:
                self.id_neighbors_dict[row['tail']][row['relation']].append(row['head'])
                
            

