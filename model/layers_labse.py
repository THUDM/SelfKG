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
import random
import faiss
import pandas as pd
import argparse

from datetime import datetime

# using labse
from transformers import AutoModel, AutoTokenizer
import torch

import collections

# Labse embedding dim
MAX_LEN = 32
LINK_NUM = 15000


def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default='zh_en')
    parser.add_argument('--model_language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')
    parser.add_argument('--model_kind_name', type=str, default="model_attention_")

    # CNN hyperparameter
    parser.add_argument('--kernel_sizes', type=tuple, default=(3, 4, 5))
    parser.add_argument('--filter_num', type=int, default=100)

    parser.add_argument('--semantic', type=bool, default=False)
    parser.add_argument('--mlp_head', type=bool, default=False)
    parser.add_argument('--neighbor', type=bool, default=False)
    parser.add_argument('--same_embedding', type=bool, default=False)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--queue_length', type=int, default=129)

    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)

    # LaBSE hyperparamter
    parser.add_argument('--max_length', type=int, default=16)
    parser.add_argument('--negative_sample_num', type=int, default=4096)
    parser.add_argument('--links_ratio', type=float, default=0.5)

    return parser.parse_args()


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


class NCESoftmaxLoss(nn.Module):
    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        # pdb.set_trace()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        # pdb.set_trace()
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

        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)
        self.logging = open(join(OUT_DIR, str(datetime.now()) + '.log'), 'w')

        self.loader_1 = DBP15kRawLoader(self.args.language).id_entity

        self.loader_2 = DBP15kLoader(self.args.language, '2').id_features_dict

        self.batch_str_1 = []
        self.batch_str_2 = []

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

        def train_link_loader(mode):
            link = {}
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), 'train.ref'), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
            return link

        self.link = link_loader(self.args.language)

        self.train_link = train_link_loader(self.args.language)

        self.dataset_sizes = {1: len(self.loader_1), 2: len(self.loader_2)}
        self.iteration = 0

        self.model = None

        if training:
            # self.writer = SummaryWriter(
            #     log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
            #     comment=self.args.time)

            self.model = LaBSEEncoder(self.args, self.device).to(self.device)

            emb_dim = EMBED_DIM

            self.iteration = 0
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-4)

            self.vector_1 = None
            self.vector_2 = None
            self.temp_links = collections.OrderedDict()
            self.inv1, self.inv2 = dict(), dict()
            self.links_ratio = self.args.links_ratio


    def save_model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language,
                               self.args.model_kind_name + str(epoch) + "_batch_size" + str(
                                   BATCH_SIZE) + "_neg_queue_len_" + str(self.args.queue_length - 1) + '.ckpt'))

    def adjust_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def evaluate(self, step, batch_id, update_temp_links=False):
        print("Evaluate at epoch {} batch {}...".format(step, batch_id))
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

        source = [_id for _id in ids_1 if _id in self.link]
        target = np.array(
            [self.inv2[self.link[_id]] for _id in source if self.link[_id] in ids_2])
        src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in self.link]

        self.vector_1 = torch.tensor(np.concatenate(tuple(vector_1), axis=0))
        vector_1 = np.concatenate(tuple(vector_1), axis=0)[src_idx, :]
        self.vector_2 = torch.tensor(np.concatenate(tuple(vector_2), axis=0))

        index = faiss.IndexFlatL2(self.vector_2.shape[1])
        index.add(np.ascontiguousarray(self.vector_2))
        D, I = index.search(np.ascontiguousarray(vector_1), 10)

        hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
        hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)

        print("#Entity:", len(source))
        print("Hit@1: ", round(hit1, 4))
        print("Hit@10:", round(hit10, 4))
        self.logging.write("Evaluate at epoch {} batch {}...\n".format(step, batch_id))
        self.logging.write('Hit@1:  {}\n'.format(round(hit1, 4)))
        self.logging.write('Hit@10: {}\n\n'.format(round(hit10, 4)))

        if update_temp_links:
            link = self.link
            D, I = index.search(np.ascontiguousarray(self.vector_1), 1)

            change = {}

            
            candidates = np.argsort(D[:, 0])[:int(D.shape[0])].tolist()  # *self.links_ratio
            change = dict((k, v) for k, v in zip(candidates, I[:, 0][candidates]))

            
            eval_links = change

            acc, tot = 0, 0
            for idx1, idx2 in eval_links.items():
                if ids_2[idx2] in link:
                    tot += 1
                    if link[ids_2[idx2]] == ids_1[idx1]:
                        acc += 1
                        self.temp_links[idx1] = idx2
            print("temp link len: ", len(self.temp_links))

            print("Candidate Acc:", round(acc / tot, 4))

    def train(self, start=0):
        # generate str batches, be cautious about the part that cannot be divided by batch size
        batch_size = BATCH_SIZE
        batch = [[], []]
        for i, (_id, _ent_name) in enumerate(self.loader_1.items()):
            batch[0].append(_id)
            batch[1].append(_ent_name)
            if (i + 1) % batch_size == 0:
                self.batch_str_1.append(batch)
                batch = [[], []]
        self.batch_str_1.append(batch)

        batch = [[], []]
        for i, (_id, _ent_name) in enumerate(self.loader_2.items()):
            batch[0].append(_id)
            batch[1].append(_ent_name)
            if (i + 1) % batch_size == 0:
                self.batch_str_2.append(batch)
                batch = [[], []]
        self.batch_str_2.append(batch)

        tot_loss = 0
        all_data_batches = []
        for epoch in range(start, self.args.epoch):
            all_data_batches.append([])

            for batch_id, (id_data_1, ent_name_1) in enumerate(self.batch_str_1):
                all_data_batches[epoch].append([1, id_data_1, ent_name_1])

            for batch_id, (id_data_2, ent_name_2) in enumerate(self.batch_str_2):
                all_data_batches[epoch].append([2, id_data_2, ent_name_2])

            random.shuffle(all_data_batches[epoch])

        self.evaluate(0, 0, True)

        for epoch in range(start, self.args.epoch):
            for batch_id, (language_id, id_data, ent_name) in tqdm(enumerate(all_data_batches[epoch])):
                self.optimizer.zero_grad()

                pos = self.model(ent_name).to(self.device)

                src_tensor = self.vector_1.to(self.device)
                tgt_tensor = self.vector_2.to(self.device)

                dist = [None, torch.LongTensor([1] * self.dataset_sizes[1]),
                        torch.LongTensor([1] * self.dataset_sizes[2])]

                if language_id == 1:
                    with torch.no_grad():

                        inv = self.inv1
                        src_pos_idx, tgt_pos_idx = list(), list()
                        src_pos_id, tgt_pos_id = list(), list()


                        for i, _id in enumerate(id_data):
                            _id = _id
                            if inv[_id] not in self.temp_links.keys(): 
                                dist[1][inv[_id]] = 0
                                src_pos_idx.append(i)
                                src_pos_id.append(inv[_id])
                            else:  # in temp link, sample from counter-dataset
                                dist[2][self.temp_links[inv[_id]]] = 0
                                tgt_pos_idx.append(i)
                                tgt_pos_id.append(self.temp_links[inv[_id]])
                    sampled_idx1 = torch.multinomial(dist[1].float().to(self.device),
                                                     self.args.negative_sample_num)
                    sampled_idx2 = torch.multinomial(dist[2].float().to(self.device),
                                                     self.args.negative_sample_num)

                    # pdb.set_trace()

                    contrastive_loss = self.model.contrastive_loss_st(src_pos_1=pos[src_pos_idx],
                                                                      tgt_pos_1=pos[tgt_pos_idx],
                                                                      src_pos_2=src_tensor[src_pos_id],
                                                                      tgt_pos_2=tgt_tensor[tgt_pos_id],
                                                                      src_neg=src_tensor[sampled_idx1],
                                                                      tgt_neg=tgt_tensor[sampled_idx2])

                else:
                    with torch.no_grad():

                        inv = self.inv2
                        src_pos_idx, tgt_pos_idx = list(), list()
                        src_pos_id, tgt_pos_id = list(), list()


                        for i, _id in enumerate(id_data):
                            _id = _id
                            # print("inv[_id]: ", inv[_id])
                            if inv[_id] not in self.temp_links.values(): 
                                dist[2][inv[_id]] = 0
                                tgt_pos_idx.append(i)
                                tgt_pos_id.append(inv[_id])
                            else:  # in temp link, sample from counter-dataset
                                #  extract key according to value
                                k = list(self.temp_links.keys())[list(self.temp_links.values()).index(inv[_id])]
                                dist[1][k] = 0
                                src_pos_idx.append(i)
                                src_pos_id.append(k)

                    sampled_idx1 = torch.multinomial(dist[1].float().to(self.device),
                                                     self.args.negative_sample_num)
                    sampled_idx2 = torch.multinomial(dist[2].float().to(self.device),
                                                     self.args.negative_sample_num)

                    # pdb.set_trace()

                    # contrastive loss
                    contrastive_loss = self.model.contrastive_loss_st(src_pos_1=pos[tgt_pos_idx],
                                                                      tgt_pos_1=pos[src_pos_idx],
                                                                      src_pos_2=tgt_tensor[tgt_pos_id],
                                                                      tgt_pos_2=src_tensor[src_pos_id],
                                                                      src_neg=tgt_tensor[sampled_idx2],
                                                                      tgt_neg=src_tensor[sampled_idx1])
                # self.writer.add_scalar(join(self.args.model, 'contrastive_loss'), contrastive_loss.data,
                #                        self.iteration)

                self.iteration += 1

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()

                if batch_id % 100 == 0 and batch_id > 0:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                contrastive_loss.detach().cpu().data / BATCH_SIZE))

            # self.save_model(self.model, epoch)
            self.evaluate(epoch, batch_id, True)
