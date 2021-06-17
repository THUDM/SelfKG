from fuzzywuzzy import fuzz
from model.layers import CNNEmbedding
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
from loader.DBP15k import DBP15kLoader
from script.preprocess.get_token import Token
from script.preprocess.deal_dataset import Mydataset

import numpy as np
from scipy.stats import pearsonr
import os
from os.path import join
from sklearn.metrics import f1_score
from settings import *


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    args = parse_options()
    '''
    setup_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader = DBP15kLoader()  # load raw data
    token = Token(loader)  # tokenize
    myset = Mydataset(token.id_features_dict)

    length = len(myset)
    train_size, test_size = int(train_prop * length), int(test_prop * length)
    train_set, test_set = torch.utils.data.random_split(myset, [train_size, test_size])

    train_loader = Data.DataLoader(
        dataset=train_set,  # torch TensorDataset format
        batch_size=len(args.batch_size),  # all test data
        shuffle=True,
        drop_last=True,
    )

    test_loader = Data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=len(args.batch_size),  # all test data
        shuffle=True,
        drop_last=True,
    )

    model = CNNEmbedding(length, 300, (3, 4, 5), 100, 1, padding_idx=-1)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)

    # start training
    for epoch in range(0, 500):
        train_accuracy = 0
        test_accuracy = 0
        pearson = 0

        for step, (batch_x, batch_y) in enumerate(sub_train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            model.zero_grad()

            tag_scores = model(batch_x)
            prediction = torch.max(tag_scores, 1)[1]
            pred_y = prediction.data.numpy()
            target_y = batch_y[:, 0].view(BATCH_SIZE).data.numpy()

            train_accuracy += (pred_y == target_y).astype(int).sum()
            ans_score = batch_y[:, 0].view(BATCH_SIZE)
            loss = loss_function(tag_scores, ans_score)

            if step == len(sub_train_loader) - 1:
                print("epoch: {}".format(epoch))
                print(' ')
                print("train accuracy = {:.2f}%".format(train_accuracy * 100 / (BATCH_SIZE * (step + 1))))

            loss.backward()
            optimizer.step()

        for step0, (batch_x, batch_y) in enumerate(sub_test_loader):
            with torch.no_grad():
                tag_scores = model(batch_x)
                ans_score = batch_y[:, 0].view(sub_test_loader.batch_size)
                loss = loss_function(tag_scores, ans_score)

                prediction = torch.max(tag_scores, 1)[1]
                pred_y = prediction.data.numpy()
                target_y = batch_y[:, 0].view(sub_test_loader.batch_size).data.numpy()
                for index, distr in enumerate(test_distr):
                    pearson += pearsonr(np.array(test_distr[index]), np.array(tag_scores[index]))[0]

                test_accuracy += (pred_y == target_y).astype(int).sum()

                if step0 == len(sub_test_loader) - 1:
                    weighted_f1 = f1_score(target_y, pred_y, average='weighted')
                    print("test accuracy = {:.2f}%".format(
                        test_accuracy * 100 / (sub_test_loader.batch_size * (step0 + 1))))
                    print("test F1 = {:.2f}%".format(weighted_f1 * 100))
                    print("coef = {:.4f}".format(pearson / len(test_distr)))
                    print('-' * 30)
