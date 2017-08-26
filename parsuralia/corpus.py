from __future__ import division

import os
import re
import shelve
import sys
import unicodedata
import json
import csv
import time
import matplotlib
import numpy as np
import nltk
import enchant
import plotly.plotly as ply
import plotly.graph_objs as pgo

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tag.perceptron import PerceptronTagger
from nltk.util import ngrams
from enchant.checker import SpellChecker

TAGS = sorted(nltk.load('help/tagsets/upenn_tagset.pickle').keys())
SEP = '|'
MAXNG = 3
tagger = PerceptronTagger()

F_WD = os.path.dirname(__file__)

F_INP = os.path.join('.', 'inputs', 'chs')
F_OUT = os.path.join('.', 'outputs', 'chs')

F_SRC = '/chs/'

class Doc:
    raws = {}
    tokenlists = {}
    datasets = {}
    statistics = {}

    name = ''
    gram = 1
    commonness = 10
    table = []

    func_list = [sum, min, np.median, np.mean, max, np.std]
    func_list_names = [a.__name__ for a in func_list]
    others = ['hapax #', 'hapax %']

    stw = stopwords.words('english')
    pkt = nltk.data.load('tokenizers/punkt/english.pickle')
    sno = nltk.stem.SnowballStemmer('english')

    def __init__(self, t, n='', g=1):
        """

        :param t: A raw text
        :param name: The name of the text: must be unique
        :param gram: window length of word tokens
        """
        self.name = n
        print "working on " + self.name
        self.gram = g

        self.raws = self.makeRaws(t.decode('utf-8'))

        # Make word and sentence lists for each version of the raw text
        for k, v in self.raws.iteritems():
            self.tokenlists.update(self.makeLists(k, v))

        self.datasets = self.makeDatasets(self.tokenlists, 't_')
        self.statistics = self.makeStatistics(self.datasets)
        self.table = self.flatten(self.statistics)
        #findAcrostics(self.tokenlists['lower_nopunct_w_'])

    @classmethod
    def fromDir(cls, dir, filename, gr=1):
        t = open(os.path.join(dir, filename), "rU").read()
        return cls(t, filename, gr)

    @classmethod
    def fromTxt(cls, t, name, gr=1):
        return cls(t, name, gr)

    def makeRaws(self, t):
        tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                            if unicodedata.category(unichr(i)).startswith('P'))
        r = {}

        r['lower_nopunct'] = t.lower().translate(tbl)

        if self.gram == 1:
            r['raw'] = t
            r['lower'] = t.lower()

        return r

    def makeLists(self, k, v):

        w = {}
        p = k + "_w_"
        w[p] = nltk.word_tokenize(v)  # [word_token,...]
        w[p + 'pos'] = nltk.tag._pos_tag(w[p], None, tagger)  # [(word, POS),...]
        w[p + 'pos_only'] = [a[1] for a in w[p + 'pos']]  # [POS...]
        w[p + 'stop'] = [t for t in w[p] if t not in Doc.stw]  # [word_token,...]
        w[p + 'stem'] = [Doc.sno.stem(t) for t in w[p]]  # [word_token,...]
        w[p + 'stop_stem'] = [Doc.sno.stem(t) for t in w[p + 'stop']]  # [word_token,...]
        w[p + 'lengths'] = [len(a) for a in w[p]]  # [length,...]

        p = k + "_s_"
        w[p] = Doc.pkt.tokenize(v)  # [sent_token,...]
        w[p + 'lengths'] = [len(nltk.word_tokenize(a)) for a in w[p]]  # [sent_lengths,...]

        return {k: [a for a in ngrams(w[k], self.gram)] for k in w.keys()}

    def makeDatasets(self, d, prefix=''):
        ds = {}
        for k, v in d.iteritems():
            # if type(v) is not list:
            # continue
            # elif type(v[0]) is not list and type(v) is list:
            #     ds[prefix + k] = nltk.FreqDist(v)
            # elif type(v[0][0]) is not list and type(v[0]) is list:
            #     ds[prefix + k] = [nltk.FreqDist(a) for a in v]
            ds[prefix + k] = nltk.FreqDist(v)

        return ds

    def makeStatistics(self, ds):
        st = {}

        for k, v in ds.iteritems():
            if len(v) == 0: continue
            st[k] = {f.__name__: f(v.values()) for f in self.func_list}

            for n, e in enumerate(v.most_common(self.commonness)):
                st[k].update({str(n) + 'th common': str(e[0])[1:100],
                              str(n) + 'th common #': e[1],
                              str(n) + 'th common %': e[1] / st[k]['sum']})

            st[k].update({'hapax #': len(v.hapaxes()),
                          'hapax %': len(v.hapaxes()) / st[k]['sum']})

        return st

    def flatten(self, s):

        rows = []
        st = self.name + SEP + str(self.gram) + '-gram' + SEP
        for k, v in s.iteritems():
            b = st + k + SEP
            for j in sorted(v.keys()):
                b += str(v[j]) + SEP
            rows.append(b)

        return rows

    def tabulate(self):
        return self.table

    @staticmethod
    def header():
        head = []
        for x in range(0, Doc.commonness):
            head.append(str(x) + 'th common')
            head.append(str(x) + 'th common #')
            head.append(str(x) + 'th common %')

        for y in Doc.func_list: head.append(y.__name__)
        for z in Doc.others: head.append(z)

        out = 'text' + SEP + 'gram' + SEP + 'dataset'
        for a in sorted(head): out += SEP + a

        return out


def findAcrostics(tokens):
    letters = [a[0][0] for a in tokens]
    grams = {}
    chkr = enchant.Dict("en_GB")

    for x in range(4, 9):
        groups = [''.join(a) for a in ngrams(letters, x)]
        validity = [(a, chkr.check(a)) for a in groups]
        for b in validity:
            if b[1]: print b


class Corpus:
    """
    Stores one set of tokenized, stemmed, n-grammed documents and information
    about those documents; Takes args docs[](list)
    """
    documents = {}
    toWrite = []
    output = ''

    def __init__(self, docs=[], names=[], ngram=1):
        for d, n in zip(docs, names):
            self.documents[n] = Doc.fromDir(d, n, ngram)
        self.write()

    def write(self):
        self.output += Doc.header() + '\n'
        for v in self.documents.itervalues():
            for r in v.tabulate(): self.output += r + '\n'


paths = [F_INP for f in os.listdir(F_INP)]
names = [f for f in os.listdir(F_INP)]
outpath = os.path.join('.', 'outputs')

#cor = []
#for a in xrange(1,MAXNG+1): cor.append(Corpus(paths,names,a))
#c = Corpus([wd+'/chs/', wd+'/chs/'],['01 - Telemachus.txt','02 - nestor.txt'])
#c = Corpus(paths[1:2], names[1:2])

#if not os.path.exists(F_OUT):
#    os.makedirs(F_OUT)

#z = open(os.path.join(F_OUT, time.strftime("%Y%m%d-%H%M%S Table.csv")),'w')
#if isinstance(c, list):
#    for o in c: z.write(o.output)
#elif isinstance(c, Corpus):
#    z.write(c.output)
#z.close()

#header = True
#for doc in c.getDocs():
#    if header: z.write(Doc.header()+'\n')
#    for row in doc.tabulate():
#        z.write(row+'\n')
#z.close()
