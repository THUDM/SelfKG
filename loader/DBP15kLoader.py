from settings import *
import pandas as pd


class DBP15kLoader(object):
    def __init__(self, language):
        assert language in ['fr_en', 'ja_en', 'zh_en']
        self.language = language
        # self.id_entity_1 = {}
        # self.id_entity_2 = {}
        self.train_set = {}
        self.test_set = {}
        self.load()

    def load(self):
        path = join(DATA_DIR, 'p_dbp15k', self.language)
        # data = pd.read_csv(join(path, 'ent_ids_1'), sep='\t', header=None)
        # data.columns = ['id', 'ent']
        # self.id_entity_1 = data.set_index('id').to_dict()['ent']
        # data = pd.read_csv(join(path, 'ent_ids_2'), sep='\t', header=None)
        # data.columns = ['id', 'ent']
        # self.id_entity_2 = data.set_index('id').to_dict()['ent']
        # data = pd.read_csv(join(path, 'sup_ent_ids'), sep='\t', header=None)
        # data.columns = ['id1', 'id2']
        # self.train_set = data.set_index('id1').to_dict()['id2']
        # data = pd.read_csv(join(path, 'ref_ent_ids'), sep='\t', header=None)
        # data.columns = ['id1', 'id2']
        # self.test_set = data.set_index('id1').to_dict()['id2']
        data = pd.read_csv(join(path, 'sup_ent'), sep='\t', header=None)
        data.columns = ['ent1', 'ent2']
        self.train_set = data.set_index('ent1').to_dict()['ent2']
        data = pd.read_csv(join(path, 'ref_ent'), sep='\t', header=None)
        data.columns = ['ent1', 'ent2']
        self.test_set = data.set_index('ent1').to_dict()['ent2']
