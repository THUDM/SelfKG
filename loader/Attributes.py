from settings import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class AttributesLoader(object):
    def __init__(self, language, doc_id):
        # whether it is in training
        self.language = language
        self.doc_id = doc_id

        self.path = join(DATA_DIR, 'DBP15K', language, "attributes")
        # load id_features
        self.id_attributes = None
        self.id_attributes_loader()

    def __getattr__(self, item):
        if item == 'corpus':
            return self.id_features['features'].tolist()

    def id_attributes_loader(self):
        data = pd.read_csv(join(self.path, 'attribute_' + self.doc_id + '.csv'), header=None)
        data.columns = ['id', 'entity', 'attri_type', 'attri_value']
        self.id_attributes = data
        print(data)

