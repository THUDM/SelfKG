from settings import *
import jieba

class ZH_id_entity():
    def __init__(self):
        self.zh_id_entity = {}
        self.load()

    def load(self):
        path = join(DATA_DIR, 'DBP15K', "zh_en")
        file = open(join(path, "cleaned_ent_ids_1.csv"),encoding='utf-8')
        for line in file:
            id_entity = line.strip().split(',')
            id = int(id_entity[0])
            entity = id_entity[1]
            word_list = jieba.lcut(entity)
            self.zh_id_entity[id] = word_list
        file.close()
    