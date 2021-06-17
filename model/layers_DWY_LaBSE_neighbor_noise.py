# coding: UTF-8
import pdb

import torch

torch.manual_seed(37)
torch.cuda.manual_seed(37)

import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np

np.random.seed(37)
import torch.optim as optim
from settings import *

import torch.utils.data as Data

from loader.DWY100KRawNeighbor import DWY100KRawNeighbors

from script.preprocess.get_token import Token
from loader.Neighbors import NeighborsLoader
from script.preprocess.neighbor_token import NeighborToken

from script.preprocess.deal_raw_dataset import MyRawdataset
import random
import faiss
import pandas as pd

import argparse

from datetime import datetime

# using labse
from transformers import *
import torch

import collections

# Labse embedding dim
MAX_LEN = 88

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--dir', type=str, default='dbp_wd')
    parser.add_argument('--model_dir', type=str, default='dbp_wd')
    parser.add_argument('--model', type=str, default='LaBSE')
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)

    parser.add_argument('--gat_num', type=int, default=1)

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--queue_length', type=int, default=64)

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.3)

    # for noise
    parser.add_argument('--lambd', type=int, default=1)

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


class MoConoid(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MoConoid, self).__init__()

        self.args = args

        self.device = torch.device(self.args.device)

        self.attn = BatchMultiHeadGraphAttention(self.device)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        )

        # self.attn_mlp2 = nn.Sequential(
        #     nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        #     nn.ReLU(),
        #     nn.Linear(LaBSE_DIM, LaBSE_DIM),
        # )

        # loss
        self.criterion = NCESoftmaxLoss(self.device)

        # batch queue
        self.batch_queue = []

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        # print("l_pos mean: ", torch.mean(l_pos, dim=0).cpu().data)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        # print("l_neg mean: ", torch.mean(l_neg, dim=0).cpu().data)
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch):
        # print("batch.shape: ", batch.shape)
        batch = batch.to(self.device)
        batch_in = batch[:, :, :LaBSE_DIM]
        adj = batch[:, :, LaBSE_DIM:]

        center = batch_in[:, 0].to(self.device)
        center_neigh = batch_in.to(self.device)
        for i in range(0, self.args.gat_num):
            center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)
        center_neigh = center_neigh[:, 0]

        if self.args.neighbor_norm:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
        if self.args.combine:
            out_hat = torch.cat((center, center_neigh), dim=1)
            out_hat = self.attn_mlp(out_hat)
            if self.args.emb_norm:
                out_hat = F.normalize(out_hat, p=2, dim=1)
        else:
            out_hat = center_neigh

        return out_hat


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, device, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, attn_dropout=0.3, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.device = device
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().to(self.device))  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        # print("attn: ", attn)
        # print("attn.shape: ", attn.shape)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Trainer(object):
    def __init__(self, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        self.seed = seed
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        loader1 = DWY100KRawNeighbors(self.args.dir, "1")
        myset1 = MyRawdataset(loader1.id_neighbors_dict, loader1.id_adj_tensor_dict)

        del loader1
        
        self.loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset1

        loader2 = DWY100KRawNeighbors(self.args.dir, "2")
        myset2 = MyRawdataset(loader2.id_neighbors_dict, loader2.id_adj_tensor_dict)
        del loader2

        self.loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset2

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode, valid=False):
            link = {}
            f = 'valid.ref' if valid else 'ref_ent_ids'
            link_data = pd.read_csv(join(join(DATA_DIR, 'DWY100K', mode), f), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.link = link_loader(self.args.dir)
        self.val_link = link_loader(self.args.dir, True)

        # queue for negative samples for 2 language sets
        self.neg_queue1 = None
        # self.neg_valid_num_queue1 = None
        self.neg_queue2 = None
        # self.neg_valid_num_queue2 = None

        self.id_list1 = []

        if training:
            # self.writer = SummaryWriter(
            #     log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
            #     comment=self.args.time)

            self.model = MoConoid(self.args, VOCAB_SIZE).to(self.device)
            self._model = MoConoid(self.args, VOCAB_SIZE).to(self.device)
            self._model.update(self.model)

            emb_dim = LaBSE_DIM
            self.iteration = 0
            self.lr = self.args.lr
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

    def save__model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_dir), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_dir,
                               "_model"
                               + "_DWY_LaBSE_neighbor"
                               + self.args.model_name
                               + "_epoch_" + str(epoch)
                               + '.ckpt'))

    def save_model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_dir), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_dir,
                               "model"
                               + "_DWY_LaBSE_neighbor"
                               + self.args.model_name
                               + "_epoch_" + str(epoch)
                               + '.ckpt'))

    def evaluate(self, step):
        print("Evaluate at epoch {}...".format(step))
        # print("Evaluate at epoch {}...".format(step), file=result_file)
        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        inverse_ids_2 = dict()
        with torch.no_grad():
            self.model.eval()
            for sample_id_1, (token_data_1, id_data_1) in tqdm(enumerate(self.eval_loader1)):
                entity_vector_1 = self.model(token_data_1).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1.squeeze().tolist())
                vector_1.append(entity_vector_1)

            for sample_id_2, (token_data_2, id_data_2) in tqdm(enumerate(self.eval_loader2)):
                entity_vector_2 = self.model(token_data_2).squeeze().detach().cpu().numpy()
                ids_2.extend(id_data_2.squeeze().tolist())
                vector_2.append(entity_vector_2)

        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx
        def cal_hit(v1, v2, link):
            source = [_id for _id in ids_1 if _id in link]
            target = np.array(
                [inverse_ids_2[link[_id]] if link[_id] in inverse_ids_2 else 9999999 for _id in source])
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
        fix_seed(self.seed)
        tot_loss = 0
        all_data_batches = []
        for batch_id, (token_data, id_data) in enumerate(self.loader1):
            all_data_batches.append([1, token_data, id_data])
        del self.loader1

        for batch_id, (token_data, id_data) in enumerate(self.loader2):
            all_data_batches.append([2, token_data, id_data])
        del self.loader2

        random.shuffle(all_data_batches)

        neg_queue = []
        # neg_valid_num = []

        print("*** Evaluate at the very beginning ***")
        self.evaluate(0)

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
            # result_file = open('result.txt','a')
            for batch_id, (dir_id, token_data, id_data) in tqdm(enumerate(all_data_batches)):
                # 后面要改的
                pos_batch = None
                if dir_id == 1:
                    with torch.no_grad():
                        if self.neg_queue1 == None:
                            self.neg_queue1 = token_data.unsqueeze(0)
                        else:
                            self.neg_queue1 = torch.cat((self.neg_queue1, token_data.unsqueeze(0)), dim = 0)

                    id_data = id_data.squeeze()

                    if self.neg_queue1.shape[0] == self.args.queue_length + 1:
                        pos_batch = self.neg_queue1[0]
                        self.neg_queue1 = self.neg_queue1[1:]
                        # add noise
                        pos_batch_lambd = pos_batch.repeat(self.args.lambd, 1)
                        neg_queue = torch.cat((self.neg_queue1, pos_batch_lambd), dim=0)
                    else:
                        continue

                else:
                    with torch.no_grad():
                        if self.neg_queue2 == None:
                            self.neg_queue2 = token_data.unsqueeze(0)
                        else:
                            self.neg_queue2 = torch.cat((self.neg_queue2, token_data.unsqueeze(0)), dim = 0)


                    if self.neg_queue2.shape[0] == self.args.queue_length + 1:
                        pos_batch = self.neg_queue2[0]
                        self.neg_queue2 = self.neg_queue2[1:]
                        # add noise
                        pos_batch_lambd = pos_batch.repeat(self.args.lambd, 1)
                        neg_queue = torch.cat((self.neg_queue2, pos_batch_lambd), dim=0)
                    else:
                        continue

                # implement queue

                self.optimizer.zero_grad()

                pos_1 = self.model(pos_batch)

                neg_value_list = []

                with torch.no_grad():
                    self._model.eval()
                    pos_2 = self._model(pos_batch)
                    neg_shape = neg_queue.shape
                    neg_queue = neg_queue.reshape(neg_shape[0]*neg_shape[1], neg_shape[2], -1)
                    neg_value = self._model(neg_queue)


                # contrastive 
                contrastive_loss = self.model.contrastive_loss(pos_1, pos_2, neg_value)

                self.iteration += 1

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()
                # update
                self._model.update(self.model)
                

                if batch_id % 200 == 0:
                    hit1_valid, hit10_valid, hit1_test, hit10_test = self.evaluate(str(epoch)+": batch "+str(batch_id))
                    if hit1_valid > best_hit1_valid:
                        best_hit1_valid = hit1_valid
                        best_hit1_valid_hit10 = hit10_valid
                        best_hit1_valid_epoch = epoch
                        record_epoch = epoch
                        record_batch_id = batch_id
                        record_hit1 = hit1_test
                        record_hit10 = hit10_test
                    if hit10_valid  > best_hit10_valid:
                        best_hit10_valid = hit10_valid
                        best_hit10_valid_hit1 = hit1_valid
                        best_hit10_valid_epoch = epoch
                        if hit1_valid == best_hit1_valid:
                            record_epoch = epoch
                            record_batch_id = batch_id
                            record_hit1 = hit1_test
                            record_hit10 = hit10_test 

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
                    print('Test @ Best Valid = {}({}) at epoch {} batch {}'.format(record_hit1, record_hit10, record_epoch, record_batch_id))

                    print('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch))
                    print('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test,best_hit10_test_hit1, best_hit10_test_epoch))
                    print("====================================")

