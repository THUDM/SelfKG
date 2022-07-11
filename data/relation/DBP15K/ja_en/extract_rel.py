import re
from os.path import join
import pdb
from tqdm import tqdm


def extract_relation(path, name, triple_path, ent_id, rel_id):
    id2enturl = {}
    with open(ent_id, 'r', encoding='utf-8') as f:
        for line in f:
            id_entity = line.strip().split('\t')
            id2enturl[int(id_entity[0])] = id_entity[1].strip()
    
    h_2_t2rel_url = {}
    with open(join(path, name), 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            h_r_t_url = line.strip().split('\t')
            h_url = h_r_t_url[0]
            try:
                r_url = h_r_t_url[1]
                t_url = h_r_t_url[2]
            except:
                continue
            if h_url in id2enturl.values() and t_url in id2enturl.values():
                if not h_url in h_2_t2rel_url.keys():
                    h_2_t2rel_url[h_url] = {}
                h_2_t2rel_url[h_url][t_url] = r_url
            # if not t_url in list(h_2_t2rel_url.keys()):
            #     h_2_t2rel_url[t_url] = {}
            # h_2_t2rel_url[t_url][h_url] = r_url

    # pdb.set_trace()

    id_rel = {}
    with open(triple_path, 'r', encoding='utf-8') as f:
        for line in f:
            h_r_t = line.strip().split('\t')
            h = int(h_r_t[0])
            r = int(h_r_t[1])
            t = int(h_r_t[2])
            try:
                r_url = h_2_t2rel_url[id2enturl[h]][id2enturl[t]]
                rel = r_url.split('/')[-1]
                id_rel[r] = rel
                print("rel: ", r)
            except:
                print("No")
                print("hurl: ", id2enturl[h])
                print("turl: ", id2enturl[t])
                print(line)
                continue
    
    
    with open("cleaned_" + rel_id, 'w') as fw:
        ids = list(id_rel.keys())
        ids.sort()
        for _id in ids:
            fw.write(str(_id) + '\t' + id_rel[_id] + '\n')


if __name__ == "__main__":
    language = "ja_en"
    path = "attributes/"
    name_1 = "ja_rel_triples"
    name_2 = "en_rel_triples"
    extract_relation(path, name_1, 'triples_1', 'ent_ids_1', 'rel_ids_1')
    extract_relation(path, name_2, 'triples_2', 'ent_ids_2', 'rel_ids_2')
