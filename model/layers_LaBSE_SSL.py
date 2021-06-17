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
from loader.DBP15k import DBP15kLoader

from script.preprocess.get_token import Token
from loader.Neighbors import NeighborsLoader
from script.preprocess.neighbor_token import NeighborToken

from script.preprocess.deal_dataset import Mydataset
import random
import faiss
import pandas as pd

import argparse

import torchtext.vocab as vocab

from script.preprocess.deal_fasttext import FastTextEmbedding

from tensorboardX import SummaryWriter
from datetime import datetime

from script.preprocess.zh_deal_fasttext import FastText_zh_Embedding

# using labse
from transformers import *
import torch

import collections

# Labse embedding dim
MAX_LEN = 88
LINK_NUM = 15000

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default='zh_en')
    parser.add_argument('--model_language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')
    # parser.add_argument('--model_kind_name', type=str, default="model_attention_")

    # CNN hyperparameter
    parser.add_argument('--kernel_sizes', type=tuple, default=(3, 4, 5))
    parser.add_argument('--filter_num', type=int, default=100)

    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--mlp_head', type=bool, default=False)
    parser.add_argument('--neighbor', type=bool, default=False)
    parser.add_argument('--same_embedding', type=bool, default=False)
    parser.add_argument('--multi', type=bool, default=False)

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=64)

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LaBSEEncoder(nn.Module):

    def __init__(self, args, device):
        super(LaBSEEncoder, self).__init__()
        self.device = device
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(join(DATA_DIR, "LaBSE"), do_lower_case=False)
        self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)
        self.criterion = NCESoftmaxLoss(self.device)

    def forward(self, batch):
        sentences = batch
        #  text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
        tok_res = self.tokenizer(sentences, add_special_tokens=True, padding='max_length', max_length=MAX_LEN)
        input_ids = torch.LongTensor([d[:MAX_LEN] for d in tok_res['input_ids']]).to(self.device)
        token_type_ids = torch.LongTensor(tok_res['token_type_ids']).to(self.device)
        attention_mask = torch.LongTensor(tok_res['attention_mask']).to(self.device)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return F.normalize(output[0][:, 1:-1, :].sum(dim=1))

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)

    def contrastive_loss_st(self, src_pos_1, tgt_pos_1, src_pos_2, tgt_pos_2, src_neg, tgt_neg):
        src_pos_1.to(self.device)
        src_pos_2.to(self.device)

        bsz_src = src_pos_1.shape[0]
        if bsz_src > 0:
            src_pos_logits = torch.bmm(src_pos_1.view(bsz_src, 1, -1), src_pos_2.view(bsz_src, -1, 1))
            src_neg_logits = torch.mm(src_pos_1.view(bsz_src, -1), src_neg.t())
            src_logits = torch.cat((src_pos_logits, src_neg_logits.unsqueeze(1)), dim=2)

        bsz_tgt = tgt_pos_1.shape[0]
        if bsz_tgt > 0:
            tgt_pos_logits = torch.bmm(tgt_pos_1.view(bsz_tgt, 1, -1), tgt_pos_2.view(bsz_tgt, -1, 1))
            tgt_neg_logits = torch.mm(tgt_pos_1.view(bsz_tgt, -1), tgt_neg.t())
            tgt_logits = torch.cat((tgt_pos_logits, tgt_neg_logits.unsqueeze(1)), dim=2)

        if bsz_src > 0 and bsz_tgt > 0:
            logits = torch.cat((src_logits, tgt_logits), dim=0)
        elif bsz_src > 0:
            logits = src_logits
        else:
            logits = tgt_logits
        
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()


class NCESoftmaxLoss(nn.Module):
    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss

class MLP_head(nn.Module):
    def __init__(self, embed_dim):
        super(MLP_head, self).__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        x = self.mlp_head(x)
        return x


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head=MULTI_HEAD_DIM, f_in=EMBED_DIM, f_out=ATTENTION_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)  # bs x n_head x n x n
        # attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Trainer(object):
    def __init__(self, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        if self.args.multi:
            self.loader1 = DBP15kRawLoader(self.args.language).id_entity
        else:  
            self.loader1 = DBP15kLoader(self.args.language, '1').id_features_dict
        self.loader2 = DBP15kLoader(self.args.language, '2').id_features_dict

        self.batch_str_1 = []
        self.batch_str_2 = []

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode, valid=False):
            link = {}
            if valid == False:
                f = 'test.ref'
            else:
                f = 'valid.ref'
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), f), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.link = link_loader(self.args.language)
        self.val_link = link_loader(self.args.language, True)

        # queue for negative samples for 2 language sets
        self.neg_queue1 = []
        self.neg_queue2 = []

        # semantic embedding queue for negative samples for 2 language sets
        self.semantic_neg_queue1 = []
        self.semantic_neg_queue2 = []

        self.id_list1 = []

        if training:
            self.writer = SummaryWriter(
                log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
                comment=self.args.time)

            self.model = LaBSEEncoder(self.args, self.device).to(self.device)
            self._model = LaBSEEncoder(self.args, self.device).to(self.device)
            self._model.update(self.model)
            self.lr = self.args.lr
            emb_dim = EMBED_DIM

            if self.args.semantic:
                emb_dim += 300

            self.mlp_head = None
            if self.args.mlp_head:
                self.mlp_head = MLP_head(emb_dim).to(self.device)

            self.iteration = 0
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

    def save_model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language,
                               "model"
                               + "_neighbor_" + str(self.args.neighbor)
                               + "_semantic_" + str(self.args.semantic)
                               + "_epoch_" + str(epoch)
                               + "_batch_size_" + str(BATCH_SIZE)
                               + "_neg_queue_len_" + str(self.args.queue_length - 1)
                               + '.ckpt'))

    def evaluate(self, step):
        print("Evaluate at epoch {}...".format(step))
        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        
        self.inv1, self.inv2 = dict(), dict()
        self.idx2entid_1, self.idx2entid_2 = dict(), dict()

        with torch.no_grad():
            self.model.eval()
            for sample_id_1, (id_data_1, ent_name_1) in tqdm(enumerate(self.batch_str_1)):      
                entity_vector_1 = self.model(ent_name_1).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1)
                vector_1.append(entity_vector_1)

            for sample_id_2, (id_data_2, ent_name_2) in tqdm(enumerate(self.batch_str_2)):
                entity_vector_2 = self.model(ent_name_2).squeeze().detach().cpu().numpy()
                ids_2.extend(id_data_2)
                vector_2.append(entity_vector_2)

        
        for idx, _id in enumerate(ids_1):
            self.inv1[_id] = idx  
        for idx, _id in enumerate(ids_2):
            self.inv2[_id] = idx
        def cal_hit(v1, v2, link):
            source = [_id for _id in ids_1 if _id in link]
            target = np.array(
                [self.inv2[link[_id]] if link[_id] in self.inv2 else 99999 for _id in source])
            src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link]
            v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
            v2 = np.concatenate(tuple(v2), axis=0)
            index = faiss.IndexFlatL2(v2.shape[1])
            index.add(np.ascontiguousarray(v2))
            D, I = index.search(np.ascontiguousarray(v1), 10)
            hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
            hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)
            print("#Entity:", len(source))
            print("Hit@1: ", round(hit1, 4))
            print("Hit@10:", round(hit10, 4))
            return hit1, hit10
        print('========Validation========')
        hit1_valid, hit10_valid = cal_hit(vector_1, vector_2, self.val_link)
        print('===========Test===========')
        hit1_test, hit10_test = cal_hit(vector_1, vector_2, self.link)
        return hit1_valid, hit10_valid, hit1_test, hit10_test

    def train(self, start=0):
        batch_size = self.args.batch_size
        batch = [[],[]]
        for i, (_id, _ent_name) in enumerate(self.loader1.items()):
            batch[0].append(_id) 
            batch[1].append(_ent_name) 
            if (i + 1) % batch_size == 0:
                self.batch_str_1.append(batch)
                batch = [[],[]]
        self.batch_str_1.append(batch)

        batch = [[],[]]
        for i, (_id, _ent_name) in enumerate(self.loader2.items()):
            batch[0].append(_id) 
            batch[1].append(_ent_name) 
            if (i + 1) % batch_size == 0:
                self.batch_str_2.append(batch)
                batch = [[],[]]
        self.batch_str_2.append(batch)

        self.evaluate(0)

        all_data_batches = []
        
        for epoch in range(start, self.args.epoch):
            all_data_batches.append([])
            for batch_id, (id_data_1, ent_name_1) in enumerate(self.batch_str_1):      
                all_data_batches[epoch].append([1, ent_name_1, id_data_1])
            for batch_id, (id_data_2, ent_name_2) in enumerate(self.batch_str_2):
                all_data_batches[epoch].append([2, ent_name_2, id_data_2])           
            random.shuffle(all_data_batches[epoch])

        neg_queue = []
        semantic_neg_queue = []

        best_hit1_valid_epoch = 0
        best_hit10_valid_epoch = 0
        best_hit1_test_epoch = 0
        best_hit10_test_epoch = 0
        best_hit1_valid = 0
        best_hit10_valid = 0
        best_hit1_valid_hit10 = 0
        best_hit10_valid_hit1 = 0
        best_hit1_test = 0
        best_hit10_test = 0
        best_hit1_test_hit10 = 0
        best_hit10_test_hit1 = 0
        for epoch in range(start, self.args.epoch):
            adjust_learning_rate(self.optimizer, epoch, self.lr)
            pos_token = None
            for batch_id, (language_id, token_data, id_data) in tqdm(enumerate(all_data_batches[epoch])):
                if language_id == 1:
                    with torch.no_grad():
                        self.neg_queue1.append(token_data)

                    if len(self.neg_queue1) == self.args.queue_length + 1:
                        pos_token = self.neg_queue1[0]
                        self.neg_queue1 = self.neg_queue1[1:]
                        neg_queue = self.neg_queue1
                    else:
                        continue

                else:
                    with torch.no_grad():
                        self.neg_queue2.append(token_data)

                    if len(self.neg_queue2) == self.args.queue_length + 1:
                        pos_token = self.neg_queue2[0]
                        self.neg_queue2 = self.neg_queue2[1:]
                        neg_queue = self.neg_queue2
                    else:
                        continue

                # implement queue

                self.optimizer.zero_grad()

                pos_1 = self.model(pos_token)

                neg_value_list = []

                with torch.no_grad():
                    self._model.eval()
                    pos_2 = self._model(pos_token)

                    # the first batch is not negative samples
                    for i, neg in enumerate(neg_queue):
                        neg_value_list.append(self._model(neg))

                neg_value = torch.cat(neg_value_list, 0)

                # contrastive loss
                contrastive_loss = self.model.contrastive_loss(pos_1, pos_2, neg_value)
                self.writer.add_scalar(join(self.args.model, 'contrastive_loss'), contrastive_loss.data,
                                       self.iteration)

                self.iteration += 1

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()

                if batch_id % 200 == 0:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                contrastive_loss.detach().cpu().data / self.args.batch_size))
                # update
                self._model.update(self.model)

            self.save_model(self.model, epoch)
            hit1_valid, hit10_valid, hit1_test, hit10_test = self.evaluate(epoch)
            if hit1_valid > best_hit1_valid:
                best_hit1_valid = hit1_valid
                best_hit1_valid_hit10 = hit10_valid
                best_hit1_valid_epoch = epoch
            if hit10_valid  > best_hit10_valid:
                best_hit10_valid = hit10_valid
                best_hit10_valid_hit1 = hit1_valid
                best_hit10_valid_epoch = epoch

            if hit1_test > best_hit1_test:
                best_hit1_test = hit1_test
                best_hit1_test_hit10 = hit10_test
                best_hit1_test_epoch = epoch
            if hit10_test  > best_hit10_test:
                best_hit10_test = hit10_test
                best_hit10_test_hit1 = hit1_test
                best_hit10_test_epoch = epoch
            print('Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10, best_hit1_valid_epoch))
            print('Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid,best_hit10_valid_hit1, best_hit10_valid_epoch))
            print('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch))
            print('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test,best_hit10_test_hit1, best_hit10_test_epoch))
            print("====================================")

