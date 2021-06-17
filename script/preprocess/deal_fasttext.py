import settings
import pandas as pd
from loader.DBP15k import DBP15kLoader
from settings import *
import torch

import torchtext.vocab as vocab


class FastTextEmbedding(object):
    fastText = vocab.FastText()
    def __init__(self, loader):
        self.loader = loader
        self.fastText = vocab.FastText()
        self.id_embedding_dict = {}

        def get_embedding():
            loader_id_features_copy = self.loader.id_features['features'].copy()
            loader_id = self.loader.id_features['id'].values.tolist()

            for i, feature in enumerate(loader_id_features_copy):              
                words = feature.strip().split(' ')

                # all 0 vector is the padding
                embed_tensor = self.fastText[words[0].lower()].view(1, FASTTEXT_DIM)
                for j, w in enumerate(words):
                    if j==0:
                        continue
                    embed_tensor = torch.cat((embed_tensor, self.fastText[w.lower()].view(1, FASTTEXT_DIM)), 0)

                self.id_embedding_dict[loader_id[i]] = embed_tensor

            # padding to the same length, use token length in settings.py
            for k, v in self.id_embedding_dict.items():
                if len(v) <= TOKEN_LEN:
                    padding = torch.cat([torch.tensor([0]*FASTTEXT_DIM)] * (TOKEN_LEN - len(v)), 0 ).view(-1, FASTTEXT_DIM)
                    self.id_embedding_dict[k] = torch.cat((v, padding), 0)
                else:
                    self.id_embedding_dict[k] = v[:TOKEN_LEN]

            return self.id_embedding_dict

        self.id_embedding_dict = get_embedding()
