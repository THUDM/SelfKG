# coding: UTF-8
import pdb

import torch

torch.manual_seed(37)
torch.cuda.manual_seed(37)

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(37)
import torch.optim as optim
from settings import *

import torch.utils.data as Data
from loader.DBP15kRawLoader import DBP15kRawLoader
from loader.DWY100K import DWY100KLoader

from loader.BWLoader import BWLoader

from script.preprocess.get_token import Token
from loader.Neighbors import NeighborsLoader
from script.preprocess.neighbor_token import NeighborToken

from script.preprocess.deal_dataset import Mydataset
import random
import faiss
import pandas as pd

import argparse

# import torchtext.vocab as vocab

from datetime import datetime

# using labse
from transformers import *
import torch

import pickle

MAX_LEN = 130

class LaBSEEncoder(nn.Module):
    def __init__(self):
        super(LaBSEEncoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(join(DATA_DIR, "LaBSE"), do_lower_case=False)
        self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)

    def forward(self, batch):
        sentences = batch
        #  text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
        tok_res = self.tokenizer(sentences, add_special_tokens=True, padding='max_length', max_length=MAX_LEN)
        input_ids = torch.LongTensor([d[:MAX_LEN] for d in tok_res['input_ids']]).to(self.device)
        token_type_ids = torch.LongTensor(tok_res['token_type_ids']).to(self.device)
        attention_mask = torch.LongTensor(tok_res['attention_mask']).to(self.device)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return F.normalize(output[0][:, 1:-1, :].sum(dim=1))


class Embedding(object):
    def __init__(self):

        self.loader1 = DBP15kRawLoader(language, '1').id_entity
        self.loader2 = DBP15kRawLoader(language, '2').id_entity

        self.model = LaBSEEncoder().to(device)

    def embedding(self):

        id_embedding_1 = {}
        for i, (_id, _ent_name) in tqdm(enumerate(self.loader1.items())):
            emb = self.model([_ent_name]).cpu().detach().numpy().tolist()
            id_embedding_1[int(_id)] = emb
        with open(join(dir_path, "raw_LaBSE_emb_1.pkl"),'wb') as f:
            pickle.dump(id_embedding_1, f)

        id_embedding_2 = {}
        for i, (_id, _ent_name) in tqdm(enumerate(self.loader2.items())):
            emb = self.model([_ent_name]).cpu().detach().numpy().tolist()
            id_embedding_2[int(_id)] = emb
        with open(join(dir_path, "raw_LaBSE_emb_2.pkl"),'wb') as f:
            pickle.dump(id_embedding_2, f)

class DWYEmbedding(object):
    def __init__(self):
 
        self.loader1 = DWY100KLoader("dbp_wd", '1').id_ent
        self.loader2 = DWY100KLoader("dbp_wd", '2').id_ent

        self.model = LaBSEEncoder().to(device)

    def embedding(self, dir_path):

        id_embedding_1 = {}
        for i, (_id, _ent_name) in enumerate(self.loader1.items()):
            emb = self.model([_ent_name]).cpu().detach().numpy().tolist()
            id_embedding_1[int(_id)] = emb
        with open(join(dir_path, "raw_LaBSE_emb_1.pkl"),'wb') as f:
            pickle.dump(id_embedding_1, f)

        id_embedding_2 = {}
        for i, (_id, _ent_name) in enumerate(self.loader2.items()):
            emb = self.model([_ent_name]).cpu().detach().numpy().tolist()
            id_embedding_2[int(_id)] = emb
        with open(join(dir_path, "raw_LaBSE_emb_2.pkl"),'wb') as f:
            pickle.dump(id_embedding_2, f)


if __name__ == "__main__":
    device = "cuda:0"
    embeder = Embedding()
    language_list = ["zh_en", "ja_en", "fr_en"]
    for language in language_list:
        dir_path = join(DATA_DIR, 'DBP15K', language)
        embeder.embedding()
    
    subset_list = ["dbp_wd", "dbp_yg"]
    dwyembeder = DWYEmbedding()
    for subset in subset_list:
        dir_path = join(DATA_DIR, 'DWY100K', subset)
        dwyembeder.embedding()
