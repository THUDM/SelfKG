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

from transformers import *


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

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--queue_length', type=int, default=129)

    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()


class LaBSEEncoder(nn.Module):
    def __init__(self, args, device):
        super(LaBSEEncoder, self).__init__()
        self.args = args
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(join(DATA_DIR, "LaBSE"), do_lower_case=False)
        self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)

        self.output_mlp = nn.Sequential(
            nn.Linear(768, 300),
            nn.ReLU()
        )

        self.criterion = NCESoftmaxLoss(self.device)

    def forward(self, batch):
        batch_list = batch.cpu().numpy().tolist()
        sentences = []
        for i, s in enumerate(batch_list):
            sent = ''
            for j, o in enumerate(s):
                sent += chr(o)
            sentences.append(sent)
        # text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or
        # `List[List[str]]` (batch of pretokenized examples).
        tok_res = self.tokenizer(sentences, add_special_tokens=True, padding='max_length', max_length=32)
        input_ids = torch.LongTensor([d[:32] for d in tok_res['input_ids']]).to(self.device)
        token_type_ids = torch.LongTensor(tok_res['token_type_ids']).to(self.device)
        attention_mask = torch.LongTensor(tok_res['attention_mask']).to(self.device)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return F.normalize(output[0][:, 1:-1, :].sum(dim=1))  # )self.output_mlp(

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

        self.id_list1 = []

        if training:
            self.writer = SummaryWriter(
                log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
                comment=self.args.time)

            self.model = LaBSEEncoder(self.args, self.device).to(self.device)
            self._model = LaBSEEncoder(self.args, self.device).to(self.device)
            self._model.update(self.model)

            emb_dim = EMBED_DIM

            if self.args.semantic:
                emb_dim += 300

            self.mlp_head = None
            if self.args.mlp_head:
                self.mlp_head = MLP_head(emb_dim).to(self.device)

            self.iteration = 0
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr)
            # self._optimizer = optim.Adam(params=self._model.parameters(), lr=self.args.lr)

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
                    entity_vector_1 = self.model(token_data_1).squeeze().detach().cpu().numpy()
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
        tot_loss = 0
        all_data_batches = []
        for epoch in range(start, self.args.epoch):
            all_data_batches.append([])
            for batch_id, (token_data, id_data) in enumerate(self.loader1):
                all_data_batches[epoch].append([1, token_data, id_data])
            for batch_id, (token_data, id_data) in enumerate(self.loader2):
                all_data_batches[epoch].append([2, token_data, id_data])
            random.shuffle(all_data_batches[epoch])

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
                sample_index = torch.cat(sample_index, dim=0).cuda(self.device).squeeze()
            for batch_id, (language_id, token_data, id_data) in tqdm(enumerate(all_data_batches[epoch])):

                self.optimizer.zero_grad()
                # self._optimizer.zero_grad()
                query = self.model(token_data)  # [batch_size, token_len]
                index = faiss.IndexFlatL2(768)
                index.add(sample_pool.detach().cpu().numpy())
                k = 50
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
                with torch.no_grad():
                    sample_token_ = torch.cat(
                        [torch.index_select(sample_token, dim=0, index=batch_index).cuda(self.device) for batch_index in
                         hard_index]) \
                        # sample_token.shape: [batch_size, k, token_len]

                    self._model.eval()
                    key_pos = self._model(token_data)
                    neg_sample = self._model(sample_token_.view(-1, TOKEN_LEN))  # [batch_size*k, token_len]
                # contrastive loss
                contrastive_loss = self.model.contrastive_loss(query, key_pos, neg_sample.view(-1, 768))

                self.writer.add_scalar(join(self.args.model, 'contrastive_loss'), contrastive_loss.data,
                                       self.iteration)

                self.iteration += 1

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()
                self._model.update(self.model)

                # self._optimizer.step()
                if batch_id % 50 == 0:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                contrastive_loss.detach().cpu().data / BATCH_SIZE))
            self.save_model(self.model, epoch)
            self.evaluate(epoch)
            print("====================================")
