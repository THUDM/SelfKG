# coding: UTF-8
import pdb
from time import *

import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from settings import *

import torch.utils.data as Data
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


def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default='zh_en')
    parser.add_argument('--model_language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=64)

     # CNN hyperparameter
    parser.add_argument('--kernel_sizes', type=tuple, default=(3, 4, 5))
    parser.add_argument('--filter_num', type=int, default=100)

    parser.add_argument('--same_embedding', type=bool, default=False)


    parser.add_argument('--center_norm', type=bool, default=False)
    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)

    parser.add_argument('--gat_num', type=int, default=1)

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


class CNNEncoder(nn.Module):
    def __init__(self, args, vocab_size, embed_dim, filter_sizes, filter_num, dropout, padding_idx=-1):
        super(CNNEncoder, self).__init__()
        self.args = args
        self.embed_dim = embed_dim

        self.filter_num = filter_num
        self.filter_sizes = filter_sizes

        # CNN encoder
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, self.filter_num, (k, self.embed_dim)) for k in self.filter_sizes])
        self.output_mlp = nn.Sequential(
            nn.Linear(filter_num * len(filter_sizes), filter_num * len(filter_sizes)),
            nn.ReLU(),
            nn.Linear(filter_num * len(filter_sizes), filter_num * len(filter_sizes)),
        )

        # batch normalization
        self.norm = nn.BatchNorm1d(self.embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), self.embed_dim)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.output_mlp(x)
        if self.args.norm:
            x = F.normalize(x, p=2, dim=1)

        return x


class CNNEmbedding(nn.Module):

    def __init__(self, args, vocab_size, embed_dim, filter_sizes, filter_num, dropout, padding_idx=-1):
        super(CNNEmbedding, self).__init__()

        self.args = args

        self.embed_dim = embed_dim

        self.CNN_encoder = CNNEncoder(args, vocab_size, embed_dim, filter_sizes, filter_num, dropout, padding_idx)
        self.CNN_encoder_neighbor = None
        if args.same_embedding:
            self.CNN_encoder_neighbor = self.CNN_encoder
        else:
            self.CNN_encoder_neighbor = CNNEncoder(args, vocab_size, embed_dim, filter_sizes, filter_num, dropout,
                                                   padding_idx)

        # initialize an embedding using tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)

    def forward(self, x, is_neighbor = False):
        x = self.embedding(x)
        if not is_neighbor:
            x = self.CNN_encoder(x)
            return x
        else:
            neighbor_shape = x.shape
            neigh = x.reshape(neighbor_shape[0]*neighbor_shape[1], neighbor_shape[2], -1)
            x_neighbor = self.CNN_encoder_neighbor(neigh)
            x_neighbor = x_neighbor.reshape(neighbor_shape[0], neighbor_shape[1], -1)
            return x_neighbor  


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


class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args

        self.device = torch.device(self.args.device)

        self.encoder = CNNEmbedding(args, vocab_size, EMBED_DIM, args.kernel_sizes, args.filter_num, args.dropout,
                                    padding_idx=padding)

        self.attn = BatchMultiHeadGraphAttention(self.device, self.args)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        )

        
        # loss
        self.criterion = NCESoftmaxLoss(self.device)

        # batch queue
        self.batch_queue = []

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch):
        batch = batch.to(self.device)
        batch_in = batch[:, :, :TOKEN_LEN]
        adj = batch[:, :, TOKEN_LEN:]

        center = self.encoder(batch_in[:, 0].to(self.device))
        center_neigh = self.encoder(batch_in.to(self.device), True)


        for i in range(0, self.args.gat_num):
            center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)
        
        center_neigh = center_neigh[:, 0]

        if self.args.center_norm:
            center = F.normalize(center, p=2, dim=1)
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
    def __init__(self, device, n_head=MULTI_HEAD_DIM, f_in=EMBED_DIM, f_out=ATTENTION_DIM, attn_dropout=0.3, bias=True):
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

        loader1 = DBP15kLoader(self.args.language, '1')  # load raw data
        token1 = Token(loader1)  # tokenize
        neigh_loader1 = NeighborsLoader("zh_en", '1')
        neigh_token1 = NeighborToken(token1, neigh_loader1)
        # {id: valid_num}
        # self.id_validnum_dict1 = neigh_token1.id_validnum_dict
        myset1 = Mydataset(neigh_token1.id_neighbors_dict, neigh_token1.id_adj_tensor_dict)  # dataset

        self.id_embedding_dict_1 = None

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

        loader2 = DBP15kLoader(self.args.language, '2')  # load raw data
        token2 = Token(loader2)  # tokenize
        neigh_loader2 = NeighborsLoader("zh_en", '2')
        neigh_token2 = NeighborToken(token2, neigh_loader2)
        # self.id_validnum_dict2 = neigh_token2.id_validnum_dict
        myset2 = Mydataset(neigh_token2.id_neighbors_dict, neigh_token2.id_adj_tensor_dict)  # dataset

        self.id_embedding_dict_2 = None

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

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode):
            link = {}
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), 'test.ref'), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.link = link_loader(self.args.language)

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

            self.model = MyEmbedder(self.args, VOCAB_SIZE).to(self.device)
            self._model = MyEmbedder(self.args, VOCAB_SIZE).to(self.device)
            self._model.update(self.model)

            emb_dim = EMBED_DIM

            self.iteration = 0
            self.lr = self.args.lr
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

    def save_model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language,
                               "model"
                               + "_neighbor_" + str(self.args.neighbor)
                               + "_epoch_" + str(epoch)
                               + "_batch_size_" + str(BATCH_SIZE)
                               + "_neg_queue_len_" + str(self.args.queue_length - 1)
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
        source = [_id for _id in ids_1 if _id in self.link]
        target = np.array(
            [inverse_ids_2[self.link[_id]] if self.link[_id] in inverse_ids_2 else 99999 for _id in source])
        src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in self.link]

        vector_1 = np.concatenate(tuple(vector_1), axis=0)[src_idx, :]
        vector_2 = np.concatenate(tuple(vector_2), axis=0)

        index = faiss.IndexFlatL2(vector_2.shape[1])
        index.add(np.ascontiguousarray(vector_2))
        D, I = index.search(np.ascontiguousarray(vector_1), 10)

        hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
        hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)

        print("#Entity:", len(source))
        print("Hit@1: ", round(hit1, 4))
        print("Hit@10:", round(hit10, 4))


    def train(self, start=0):
        fix_seed(self.seed)
        tot_loss = 0
        all_data_batches = []
        for epoch in range(start, self.args.epoch):
            all_data_batches.append([])
            for batch_id, (token_data, id_data) in enumerate(self.loader1):
                all_data_batches[epoch].append([1, token_data, id_data])
            for batch_id, (token_data, id_data) in enumerate(self.loader2):
                all_data_batches[epoch].append([2, token_data, id_data])
            random.shuffle(all_data_batches[epoch])

        print("*** Evaluate at the very beginning ***")
        self.evaluate(0)

        neg_queue = []

        for epoch in range(start, self.args.epoch):
            adjust_learning_rate(self.optimizer, epoch, self.lr)
            for batch_id, (language_id, token_data, id_data) in tqdm(enumerate(all_data_batches[epoch])):
                pos_batch = None
                if language_id == 1:
                    with torch.no_grad():
                        if self.neg_queue1 == None:
                            self.neg_queue1 = token_data.unsqueeze(0)
                        else:
                            self.neg_queue1 = torch.cat((self.neg_queue1, token_data.unsqueeze(0)), dim = 0)

                    id_data = id_data.squeeze()

                    if self.neg_queue1.shape[0] == self.args.queue_length + 1:
                        pos_batch = self.neg_queue1[0]
                        self.neg_queue1 = self.neg_queue1[1:]
                        neg_queue = self.neg_queue1
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
                        neg_queue = self.neg_queue2
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

                if batch_id % 200 == 0:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                contrastive_loss.detach().cpu().data / BATCH_SIZE))

                # update
                self._model.update(self.model)
                
                # pdb.set_trace()


            self.evaluate(epoch)
            print("====================================")
