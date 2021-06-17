from settings import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


class DBP15kLoader(object):
    def __init__(self, language, feature_id='both'):
        # whether it is in training
        self.language = language
        self.feature_id = feature_id

        self.path = join(DATA_DIR, 'DBP15K', language)
        # load id_features
        self.id_features, self.id_datasets, self.datasets_id = None, dict(), list()
        self.id_feature_loader()
        self.id_features_dict = OrderedDict(
            (self.id_features['id'][i], self.id_features['features'][i]) for i in range(len(self.id_features['id'])))
        self.link = self.link_loader()

    def __getattr__(self, item):
        if item == 'corpus':
            return self.id_features['features'].tolist()

    def link_loader(self):
        link = {}
        link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', self.language), 'ref_ent_ids'), sep='\t', header=None)
        link_data.columns = ['entity1', 'entity2']
        entity1_id = link_data['entity1'].values.tolist()
        entity2_id = link_data['entity2'].values.tolist()
        for i, _ in enumerate(entity1_id):
            link[entity1_id[i]] = entity2_id[i]
            link[entity2_id[i]] = entity1_id[i]
        return link

    def id_feature_loader(self):
        data = []
        if self.feature_id == 'both':
            for i in range(2):
                data.append(pd.read_csv(join(self.path, 'id_features_{}'.format(i + 1)), sep='\t', header=None))
                data[i].columns = ['id', 'features']
                for idx in data[i]['id']:
                    self.id_datasets[idx] = i
                self.datasets_id.append(data[i]['id'].tolist())
        else:
            data.append(pd.read_csv(join(self.path, 'id_features_' + self.feature_id), sep='\t', header=None))
            data[0].columns = ['id', 'features']
            # actually not used
            for idx in data[0]['id']:
                self.id_datasets[idx] = 0
            self.datasets_id.append(data[0]['id'].tolist())

        self.id_features = pd.concat(data)
