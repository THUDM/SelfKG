import settings
import pandas as pd
from loader.DBP15k import DBP15kLoader
from settings import *
import numpy as np


class Token(object):
    def __init__(self, loader):
        self.loader = loader
        self.embedding_list = []

        def get_token():
            id_features_dict = {}
            loader_id_features_copy = self.loader.id_features['features'].copy()
            loader_id = self.loader.id_features['id'].values.tolist()

            # max length of feature vector
            # max_len = 0
            for i, feature in enumerate(loader_id_features_copy):

                _feature = feature.strip()
                token_list = []

                # ord(' ')=32 is the padding
                for ch in _feature:
                    token_list.append(ord(ch))

                # if max_len < len(token_list):
                #     max_len = len(token_list)

                id_features_dict[loader_id[i]] = token_list

            # padding to the same length, use token length in settings.py
            len_list = []
            for k, v in id_features_dict.items():
                len_list.append(len(v))
                if len(v) <= TOKEN_LEN:
                    id_features_dict[k] = v + [ord(' ')] * (TOKEN_LEN - len(v))
                else:
                    id_features_dict[k] = v[:TOKEN_LEN]

            # print("np.percentile(len_list, 90): ", np.percentile(len_list, 90))
            return id_features_dict

        self.id_features_dict = get_token()
