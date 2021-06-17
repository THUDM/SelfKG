# coding: UTF-8
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
from loader.DBP15kLoader import DBP15kLoader
import random
import faiss
import argparse
from tensorboardX import SummaryWriter
from datetime import datetime
# using labse
from transformers import *
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
    parser.add_argument('--queue_length', type=int, default=64)

    parser.add_argument('--t', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)

    # LaBSE hyperparamter
    parser.add_argument('--max_length', type=int, default=16)
    parser.add_argument('--negative_sample_num', type=int, default=4096)
    parser.add_argument('--links_ratio', type=float, default=0.5)

    return parser.parse_args()


class LaBSEEncoder(nn.Module):
    def __init__(self, args, device):
        super(LaBSEEncoder, self).__init__()
        self.args = args
        self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(join(DATA_DIR, "LaBSE"), do_lower_case=False)
        # self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.output_mlp = nn.Sequential(
            nn.Linear(768, 300),
            nn.ReLU()
        )

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


class Trainer(object):
    def __init__(self, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)
        self.loader = DBP15kLoader(self.args.language)
        self.train_set = self.loader.train_set
        self.test_set = self.loader.test_set
        self.train_batches = []
        self.test_batches = []

        self.device = torch.device(self.args.device)

        self.model = None
        self.iteration = 0

        # queue for negative samples for 2 language sets
        self.neg_queue1 = []
        self.neg_queue2 = []

        if training:
            self.writer = SummaryWriter(
                log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
                comment=self.args.time)

            self.model = LaBSEEncoder(self.args, self.device).to(self.device)
            self._model = LaBSEEncoder(self.args, self.device).to(self.device)
            self._model.update(self.model)
            self.iteration = 0
            self.optimizer = AdamW(params=self.model.parameters(), lr=self.args.lr)

    def make_batches(self):
        batch = [[], []]
        for i, (ent1, ent2) in enumerate(self.train_set.items()):
            batch[0].append(ent1)
            batch[1].append(ent2)
            if (i + 1) % BATCH_SIZE == 0:
                self.train_batches.append(batch)
                batch = [[], []]
        self.train_batches.append(batch)
        batch = [[], []]
        for i, (ent1, ent2) in enumerate(self.test_set.items()):
            batch[0].append(ent1)
            batch[1].append(ent2)
            if (i + 1) % BATCH_SIZE == 0:
                self.test_batches.append(batch)
                batch = [[], []]
        self.test_batches.append(batch)

    def evaluate(self, step):
        print("Evaluate at epoch {}...".format(step))
        random.shuffle(self.test_batches)
        embeddings_1, embeddings_2 = [], []
        for i, (batch_ent_1, batch_ent_2) in tqdm(enumerate(self.test_batches)):
            with torch.no_grad():
                self.model.eval()
                # self._model.eval()
                embeddings_1.append(self.model(batch_ent_1).detach().cpu().numpy())
                embeddings_2.append(self.model(batch_ent_2).detach().cpu().numpy())
        embeddings_1 = np.concatenate(embeddings_1)
        embeddings_2 = np.concatenate(embeddings_2)
        index = faiss.IndexFlatL2(embeddings_1.shape[1])
        index.add(np.ascontiguousarray(embeddings_2))
        D, I = index.search(np.ascontiguousarray(embeddings_1), 10)
        hit1 = 0.
        hit10 = 0.
        for i in range(I.shape[0]):
            if I[i][0] == i:
                hit1 += 1
            if i in I[i]:
                hit10 += 1
        hit1 /= I.shape[0]
        hit10 /= I.shape[0]

        print("#Entity:", I.shape[0])
        print("Hit@1: ", round(hit1, 4))
        print("Hit@10:", round(hit10, 4))

    def evaluate_all(self, step):
        random.shuffle(self.test_batches)
        embeddings_1, embeddings_2 = [], []
        for i, (batch_ent_1, batch_ent_2) in tqdm(enumerate(self.test_batches)):
            with torch.no_grad():
                self.model.eval()
                # self._model.eval()
                embeddings_1.append(self.model(batch_ent_1).detach().cpu().numpy())
                embeddings_2.append(self.model(batch_ent_2).detach().cpu().numpy())

        for i, (batch_ent_1, batch_ent_2) in tqdm(enumerate(self.train_batches)):
            with torch.no_grad():
                self.model.eval()
                # self._model.eval()
                embeddings_1.append(self.model(batch_ent_1).detach().cpu().numpy())
                embeddings_2.append(self.model(batch_ent_2).detach().cpu().numpy())
        embeddings_1 = np.concatenate(embeddings_1)
        embeddings_2 = np.concatenate(embeddings_2)
        index = faiss.IndexFlatL2(embeddings_1.shape[1])
        index.add(np.ascontiguousarray(embeddings_2))
        D, I = index.search(np.ascontiguousarray(embeddings_1), 10)
        hit1 = 0.
        hit10 = 0.
        for i in range(I.shape[0]):
            if I[i][0] == i:
                hit1 += 1
            if i in I[i]:
                hit10 += 1
        hit1 /= I.shape[0]
        hit10 /= I.shape[0]

        print("#Entity:", I.shape[0])
        print("Hit@1: ", round(hit1, 4))
        print("Hit@10:", round(hit10, 4))

    def train(self, start=0):
        self.make_batches()
        self.evaluate_all(0)
        random.shuffle(self.train_batches)
        neg_queue = []
        for epoch in range(start, self.args.epoch):
            for batch_id, (batch_ent_1, batch_ent_2) in tqdm(enumerate(self.train_batches)):
                self.optimizer.zero_grad()
                if not neg_queue:
                    # empty queue, enqueue first batch as negative samples and do nothing
                    neg_queue.append(batch_ent_2)
                    continue
                else:
                    # not empty
                    pos_1 = self.model(batch_ent_1).cuda(self.device)
                    neg_value_list = []
                    with torch.no_grad():
                        self._model.eval()
                        pos_2 = self._model(batch_ent_2).cuda(self.device)
                        for neg in neg_queue:
                            neg_value_list.append(self._model(neg).cuda(self.device))
                    neg_value = torch.cat(neg_value_list, dim=0)
                    loss = self.model.contrastive_loss(pos_1, pos_2, neg_value)
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    neg_queue.append(batch_ent_2)
                    if len(neg_queue) == self.args.queue_length + 1:
                        neg_queue = neg_queue[1:]
                self.iteration += 1
                self._model.update(self.model)
                if batch_id % 50 == 0:
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                loss.detach().cpu().data / BATCH_SIZE))
            self.evaluate(epoch)
