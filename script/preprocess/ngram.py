from settings import *
from loader.DBP15k import DBP15kLoader

import re
from collections import Counter


class NgramIndex(object):
    def __init__(self, lo=1, hi=3, split_non_alphabet=False):
        self.lo, self.hi = lo, hi
        self.split_non_alphabet = split_non_alphabet
        self.pat_non_alphabetic = re.compile(r'[^a-z]*?')
        self.pat_numerical = re.compile(r'[0-9]*?')
        self.index = defaultdict(lambda: 0)

    def ngram(self, string):
        assert ' ' not in string
        ngrams = list()
        string = string.lower()
        if self.split_non_alphabet:
            _segs = re.split(self.pat_non_alphabetic, string)
            segs = list()
            for seg in _segs:
                seg = re.split(self.pat_numerical, seg)
                segs.extend(seg)
        else:
            segs = string.split(' ')
        n = self.hi
        for seg in segs:
            while len(seg) >= n >= self.lo:
                for i in range(len(seg) - n + 1):
                    ngrams.append(seg[i:i + n])
        return Counter(ngrams)

    def __call__(self, texts: [str]):
        for line in tqdm(texts):
            line = line.split(' ')
            ngram_counter = self.ngram(line)
            i = 1


if __name__ == '__main__':
    indexer = NgramIndex()
    loader = DBP15kLoader()
    indexer(loader.corpus)
