import settings
import pandas as pd
from settings import *
import torch

import torchtext.vocab as vocab

from script.preprocess.get_zh_id_entity import ZH_id_entity 


class FastText_zh_Embedding(object):
    fastText = vocab.FastText()
    def __init__(self):
        self.fastText = vocab.FastText()
        self.id_wordList = ZH_id_entity().zh_id_entity
        self.id_embedding_dict = {}

        def get_embedding():
            for id, word_list in self.id_wordList.items():              
                # all 0 vector is the padding
                embed_tensor = self.fastText[word_list[0]].view(1, FASTTEXT_DIM)
                for j, w in enumerate(word_list):
                    if j==0:
                        continue
                    embed_tensor = torch.cat((embed_tensor, self.fastText[w].view(1, FASTTEXT_DIM)), 0)

                self.id_embedding_dict[id] = embed_tensor

            # padding to the same length, use token length in settings.py
            for k, v in self.id_embedding_dict.items():
                if len(v) <= TOKEN_LEN:
                    padding = torch.cat([torch.tensor([0]*FASTTEXT_DIM)] * (TOKEN_LEN - len(v)), 0 ).view(-1, FASTTEXT_DIM)
                    self.id_embedding_dict[k] = torch.cat((v, padding), 0)
                else:
                    self.id_embedding_dict[k] = v[:TOKEN_LEN]

            return self.id_embedding_dict

        self.id_embedding_dict = get_embedding()
