from settings import *
import csv


class DBP15kRawLoader():
    def __init__(self, language="zh_en"):
        self.language = language
        self.id_entity = {}
        self.load()

    def load(self):
        path = join(DATA_DIR, 'DBP15K', self.language)
        with open(join(path, "cleaned_ent_ids_1"), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split('\t')
                id = int(l[0])
                entity = str(l[1])
                self.id_entity[id] = entity
    