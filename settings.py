import os
from os.path import abspath, dirname, join, exists
from collections import defaultdict
import json
import codecs
import csv
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch

def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


PROJ_DIR = abspath(dirname(__file__))
LINK_DIR = join(PROJ_DIR, 'link')
CLIENT_DIR = join(PROJ_DIR, 'client')
DATA_DIR = join(PROJ_DIR, 'data')
RAW_DATA_DIR = join(DATA_DIR, 'raw_data')
FUZZY_DIR = join(DATA_DIR, 'fuzzy')
CANDIDATE_DIR = join(PROJ_DIR, 'candidates')
os.makedirs(DATA_DIR, exist_ok=True)
OUT_DIR = join(PROJ_DIR, 'out')
EVAL_DIR = join(PROJ_DIR, 'evaluate')
os.makedirs(OUT_DIR, exist_ok=True)

TOKEN_LEN = 50
VOCAB_SIZE = 100000
LaBSE_DIM = 768
EMBED_DIM = 300
BATCH_SIZE = 96
FASTTEXT_DIM = 300
NEIGHBOR_SIZE = 20 
ATTENTION_DIM = 300
MULTI_HEAD_DIM = 1

LINK_LEN = 15000

# directory for datasets
EXPAND_DIR = join(DATA_DIR, 'expand')

# split proportion
train_prop = 1
test_prop = 1 - train_prop


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

json_dict = defaultdict(list)
e = JSONEncoder()

def get_all_jsons():
    json_dict.clear()
    for path, dir_list, file_list in os.walk(PROJ_DIR):
        for file in file_list:
            if file.endswith('.json'):
                json_dict[file].append(join(path, file))

get_all_jsons()

def read_json(filename, dir=None) -> json:
    get_all_jsons()
    if not filename.endswith('.json'):
        filename = filename + '.json'
    files = json_dict.get(filename)
    if files is None or len(files) == 0:
        raise RuntimeError("\nread_json_error: [{}] does not exist.".format(filename))
    elif len(files) > 1:
        message = "\n"
        for file in files:
            if dir and file.endswith(join(dir, filename)):
                return json.load(codecs.open(file, 'r', 'utf-8'))
            message = message + file + "\n"
        raise RuntimeError(message + "read_json_error: duplicated [{}].".format(filename))
    else:
        return json.load(codecs.open(files[0], 'r', 'utf-8'))


def write_json(data, path, filename, overwrite=False, indent=None, jsonify=False):
    if not overwrite and filename in json_dict:
        for file in json_dict[filename]:
            if file == join(path, filename):
                print("\nwrite_json_error: not allowed overwrite on [{}]".format(filename))
                print("Do you want to overwrite on the file? (y/n)\n")
                overwrite = input()
                if overwrite != 'y' and overwrite != 'Y':
                    raise RuntimeError("The user terminate the write process.")
    if jsonify:
        data = json.loads(e.encode(data))
    json.dump(data, codecs.open(join(path, filename), 'w', 'utf-8'), ensure_ascii=False, indent=indent)
    get_all_jsons()


def read_csv(path):
    f = codecs.open(path, 'r', 'utf-8')
    csv_reader = csv.reader(f)
    for row in csv_reader:
        yield row


# save in binary
def store_obj(dir, fname, obj):
    f = open(join(dir, fname), 'wb')
    pickle.dump(obj, f)
    f.close()


# load in binary
def load_obj(dir, fname):
    f = open(join(dir, fname), 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

