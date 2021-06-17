from settings import *
import pandas as pd
from os.path import abspath, dirname, join, exists
from collections import OrderedDict


class DWY100KLoader(object):
    
    def __init__(self, dataset_name='dbp_wd', set_id='1'):
        self.root_path = join(DATA_DIR, 'DWY100K', dataset_name)
        self.set_id = set_id
        self.id_ent = None
        self.id_ent_loader()


    def id_ent_loader(self):
        data = pd.read_csv(join(self.root_path, 'id_ent_' + self.set_id), sep='\t', header=None)
        data.columns = ['id', 'features']
        
        self.id_ent = OrderedDict(
            (data['id'][i], str(data['features'][i]).lower()) for i in range(len(data['id'])))

