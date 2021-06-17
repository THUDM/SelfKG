# coding: UTF-8
import torch
import torch.nn as nn
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
    parser.add_argument('--model', type=str, default='CNN')
    # parser.add_argument('--model_kind_name', type=str, default="model_attention_")

    # CNN hyperparameter
    parser.add_argument('--kernel_sizes', type=tuple, default=(3, 4, 5))
    parser.add_argument('--filter_num', type=int, default=100)

    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--mlp_head', type=bool, default=False)
    parser.add_argument('--neighbor', type=bool, default=False)
    parser.add_argument('--same_embedding', type=bool, default=False)

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--queue_length', type=int, default=129)

    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()


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
        # x = F.normalize(x, p=2, dim=1)

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

    def forward(self, x):
        x = self.embedding(x)
        if self.args.neighbor:
            x_hat = self.CNN_encoder(x[:, 0]).unsqueeze(1)
            for i in range(1, x.size(1)):
                x_hat = torch.cat((x_hat, self.CNN_encoder_neighbor(x[:, i]).unsqueeze(1)), dim=1)
            return x_hat
        x = self.CNN_encoder(x)
        return x


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

        # self.encoder_semantic = CNNEncoder(vocab_size, EMBED_DIM, (3, 4, 5), 100, 1, padding_idx=padding)
        self.encoder = CNNEmbedding(args, vocab_size, EMBED_DIM, args.kernel_sizes, args.filter_num, args.dropout,
                                    padding_idx=padding)

        self.attn = BatchMultiHeadGraphAttention()
        self.encoder_semantic = CNNEncoder(args, vocab_size, EMBED_DIM, args.kernel_sizes, args.filter_num,
                                           args.dropout, padding_idx=padding)

        self.attn_mlp = nn.Sequential(
            nn.Linear(ATTENTION_DIM * (NEIGHBOR_SIZE - 1), ATTENTION_DIM * (NEIGHBOR_SIZE - 1) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(ATTENTION_DIM * (NEIGHBOR_SIZE - 1) // 2, EMBED_DIM),
        )

        self.attn_mlp2 = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, (EMBED_DIM * 3) // 2),
            nn.ReLU(inplace=True),
            nn.Linear((EMBED_DIM * 3) // 2, EMBED_DIM),
        )

        # MLP to merge the two embeddings into the same vector space
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM + 300, EMBED_DIM + 300),
            nn.ReLU(inplace=True),
            nn.Linear(EMBED_DIM + 300, EMBED_DIM + 300),
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

    def margin_contrastzive_loss(self, pos_1, pos_2, neg_value, margin):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        l1_loss = - F.l1_loss(l_pos, l_neg, reduction='none')
        m = torch.where(l1_loss + float(margin) > 0, l1_loss + float(margin),
                        torch.zeros_like(l1_loss, device=self.device))
        return torch.mean(m)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch, batch_semantic=None):
        out = self.encoder(batch.to(self.device))
        if self.args.semantic:
            out_semantic = self.encoder_semantic(batch_semantic.to(self.device))
            out_semantic = F.normalize(out_semantic, p=1, dim=1)
            out = torch.cat((out, out_semantic), 1)
            out = self.mlp(out)
        if self.args.neighbor:
            out_n = self.attn(out).squeeze(1)
            out_neigh = out_n[:, 1]
            for i in range(2, out_n.size(1)):
                out_neigh = torch.cat((out_neigh, out_n[:, i]), dim=1)
            out_neigh = self.attn_mlp(out_neigh)
            out_hat = torch.cat((out[:, 0], out_neigh), dim=1)
            out_hat = self.attn_mlp2(out_hat)
            return out_hat
        return out


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
    def __init__(self, training=True, seed=2020):
        # # Set the random seed manually for reproducibility.
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        loader1 = DBP15kLoader(self.args.language, '1')  # load raw data
        token1 = Token(loader1)  # tokenize
        if self.args.neighbor:
            neigh_loader1 = NeighborsLoader("zh_en", '1')
            neigh_token1 = NeighborToken(token1, neigh_loader1)
            myset1 = Mydataset(neigh_token1.id_neighbors_dict)  # dataset
        else:
            myset1 = Mydataset(token1.id_features_dict)  # dataset

        self.id_embedding_dict_1 = None

        self.loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        loader2 = DBP15kLoader(self.args.language, '2')  # load raw data
        token2 = Token(loader2)  # tokenize
        if self.args.neighbor:
            neigh_loader2 = NeighborsLoader("zh_en", '2')
            neigh_token2 = NeighborToken(token2, neigh_loader2)
            myset2 = Mydataset(neigh_token2.id_neighbors_dict)  # dataset
        else:
            myset2 = Mydataset(token2.id_features_dict)  # dataset

        self.id_embedding_dict_2 = None

        self.loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        if self.args.semantic:
            self.id_embedding_dict_1 = FastText_zh_Embedding().id_embedding_dict  # embedding vectors
            self.id_embedding_dict_2 = FastTextEmbedding(loader2).id_embedding_dict  # embedding vectors

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode):
            link = {}
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), 'ref_ent_ids'), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.link = link_loader(self.args.language)

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

            self.model = MoConoid(self.args, VOCAB_SIZE).to(self.device)
            self._model = MoConoid(self.args, VOCAB_SIZE).to(self.device)
            print(self.model.encoder.embedding.state_dict())
            print(torch.load(join(PROJ_DIR, 'saved_embeddings', 'model.pt')).state_dict())
            # self.model.encoder.embedding.from_pretrained(torch.load(join(PROJ_DIR, 'saved_embeddings', 'model.pt')).weight)
            print(self.model.encoder.embedding.state_dict())

            # self._model.encoder.embedding.from_pretrained(torch.load(join(PROJ_DIR, 'saved_embeddings', '_model.pt')).weight)
            # self._model.update(self.model)

            emb_dim = EMBED_DIM

            if self.args.semantic:
                emb_dim += 300

            self.mlp_head = None
            if self.args.mlp_head:
                self.mlp_head = MLP_head(emb_dim).to(self.device)

            self.iteration = 0
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr)
            self._optimizer = optim.Adam(params=self._model.parameters(), lr=self.args.lr)

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
        inverse_ids_2 = dict()
        with torch.no_grad():
            self.model.eval()
            for sample_id_1, (token_data_1, id_data_1) in tqdm(enumerate(self.loader1)):
                if self.args.semantic:
                    semantic_batch_1 = self.id_embedding_dict_1[id_data_1[0].item()].unsqueeze(0)

                    for s, id in enumerate(id_data_1):
                        if s == 0:
                            continue
                        semantic_batch_1 = torch.cat(
                            (semantic_batch_1, self.id_embedding_dict_1[id.item()].unsqueeze(0)), 0)

                    entity_vector_1 = self.model(token_data_1,
                                                 semantic_batch_1).squeeze().detach().cpu().numpy()
                else:
                    entity_vector_1 = self.model(token_data_1,
                                                 None).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1.squeeze().tolist())
                vector_1.append(entity_vector_1)

            for sample_id_2, (token_data_2, id_data_2) in tqdm(enumerate(self.loader2)):
                if self.args.semantic:
                    semantic_batch_2 = self.id_embedding_dict_2[id_data_2[0].item()].unsqueeze(0)

                    for s, id in enumerate(id_data_2):
                        if s == 0:
                            continue
                        semantic_batch_2 = torch.cat(
                            (semantic_batch_2, self.id_embedding_dict_2[id.item()].unsqueeze(0)), 0)

                    entity_vector_2 = self.model(token_data_2,
                                                 semantic_batch_2).squeeze().detach().cpu().numpy()

                else:
                    entity_vector_2 = self.model(token_data_2,
                                                 None).squeeze().detach().cpu().numpy()
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
        tot_loss = 0
        is_first1 = True
        is_first2 = True
        all_data_batches = []
        for epoch in range(start, self.args.epoch):
            all_data_batches.append([])
            for batch_id, (token_data, id_data) in enumerate(self.loader1):
                all_data_batches[epoch].append([1, token_data, id_data])
            for batch_id, (token_data, id_data) in enumerate(self.loader2):
                all_data_batches[epoch].append([2, token_data, id_data])
            random.shuffle(all_data_batches[epoch])

        neg_queue = []
        semantic_neg_queue = []

        for epoch in range(start, self.args.epoch):
            sample_pool, sample_token, sample_index = list(), list(), list()
            with torch.no_grad():
                for batch_data in all_data_batches[epoch]:
                    sample_pool.append(self._model(batch_data[1]))
                    sample_token.append(batch_data[1])
                    sample_index.append(batch_data[2])
                sample_pool = torch.cat(sample_pool, dim=0).cuda(self.device)
                sample_token = torch.cat(sample_token, dim=0).cuda(self.device)
                sample_index = torch.cat(sample_index, dim=0).cuda(self.device)
            for batch_id, (language_id, token_data, id_data) in tqdm(enumerate(all_data_batches[epoch])):

                self.optimizer.zero_grad()
                self._optimizer.zero_grad()
                query = self.model(token_data)  # [batch_size, token_len]
                key_pos = self._model(token_data)
                index = faiss.IndexFlatL2(EMBED_DIM)
                index.add(sample_pool.detach().cpu().numpy())
                k = 1000
                D, I = index.search(query.detach().cpu().numpy(), k)  # I.shape: [batch_size, k]
                hard_index = torch.from_numpy(I).cuda(self.device)  # [batch_size, k]
                index_list = list()
                id_data = id_data.to(self.device)
                sample_index_ = torch.stack(
                    [torch.index_select(sample_index, dim=0, index=batch_index).cuda(self.device)
                     for batch_index in hard_index], dim=0).squeeze()  # sample_index_.shape: [batch_size, k]

                for i, b in enumerate(sample_index_):

                    if id_data[i] in b:
                        index_list.append(hard_index[i, id_data[i] != hard_index[i]])
                    else:
                        index_list.append(hard_index[i, :-1])
                hard_index = torch.cat(index_list, dim=0).cuda(self.device)


                sample_token_ = torch.cat(
                    [torch.index_select(sample_token, dim=0, index=batch_index).cuda(self.device)
                     for batch_index in hard_index])  # sample_token_.shape: [batch_size*k, token_len]


                neg_sample = self._model(sample_token_.view(-1, TOKEN_LEN))  # [batch_size*k, token_len]
                # contrastive loss
                contrastive_loss = self.model.contrastive_loss(query, key_pos, neg_sample.view(-1, EMBED_DIM))

                # contrastive_loss = self.model.margin_contrastive_loss(pos_1, pos_2, neg_value, 1)
                self.writer.add_scalar(join(self.args.model, 'contrastive_loss'), contrastive_loss.data,
                                       self.iteration)

                self.iteration += 1

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()
                self._optimizer.step()
                if batch_id % 50 == 0:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                contrastive_loss.detach().cpu().data / BATCH_SIZE))

            self.save_model(self.model, epoch)
            self.evaluate(epoch)
