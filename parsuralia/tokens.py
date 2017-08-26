import collections
import copy
import gc
import glob
import itertools
import nltk
import nltk.util
from nltk.util import ngrams
import nltk.tokenize
import numpy
import os
import re
import string
import time
import pickle
import platform
from nltk.corpus import stopwords

if platform.system() == 'Darwin':
    win = False
else:
    win = True

class Corpus:
    """
    Stores one set of tokenized, stemmed, n-grammed documents and information about those documents; Takes args docs[](list)
    """

    def __init__(self, d=[], names=[], ngram=0, stemmer=0):
        self.documents = d
        self.rowHeads = names
        self.nGramParam = str(ngram)
        self.stemmerParam = stemmerModules[stemmer][1]
        self.toWrite = []

        self.textLength = []
        self.lengthCoefficient = []
        self.totalDistinctTokens = []
        self.uniqueTokens = []
        self.meanTokenUniqueness = []
        self.stdDevTokenUniqueness = []
        self.meanTokenFrequency = []
        self.stdDevTokenFrequency = []
        self.top300TokenFrequency = []
        self.stdDevTop300TokenFrequency = []


    nGramParam = ''
    stemmerParam = ''
    documents = []
    rowHeads = []
    firstRow = ['Text Length,', 'Normalization Factor,', 'Total Distinct Tokens,', 'Total Unique Tokens,',
                'Mean Token Commonness (0-17),', 'stdDev of Mean Token Commonness,', 'Mean Token Frequency,',
                'stdDev of Mean Token Frequency,', 'Top 300 Mean Token Frequency,',
                'stdDev of Top 300 Mean Token Frequency,']
    toWrite = []
    textLength = []
    lengthCoefficient = []
    totalDistinctTokens = []
    uniqueTokens = []
    meanTokenUniqueness = []
    stdDevTokenUniqueness = []
    meanTokenFrequency = []
    stdDevTokenFrequency = []
    top300TokenFrequency = []
    stdDevTop300TokenFrequency = []


    def process(self):
        # append to corpusStats string

        #generate textLength and lengthCoefficient
        self.textLength = [len(text) for text in self.documents]

        self.lengthCoefficient = [float((len(text))) / max(self.textLength) for text in self.documents]

        #generate totalDistinctTokens
        self.totalDistinctTokens = [len(set(text)) for text in self.documents]

        #iterative chapter measurements

        for i, title in enumerate(self.documents):
            # generate temp set with all but chapter being measured
            tempList = []
            tempList = self.documents[:]
            del tempList[i]

            #generate uniqueTokens
            self.uniqueTokens.append(len(set(title).difference(*tempList)))

            #generate meanTokenUniqueness and stdDevTokenUniqueness
            uniquenessData = dict.fromkeys(collections.Counter(title).iterkeys(), 0)

            for text in tempList:
                for word in set(text):
                    if word in uniquenessData:
                        uniquenessData[word] += 1

            self.meanTokenUniqueness.append(numpy.mean(uniquenessData.values()))
            self.stdDevTokenUniqueness.append(numpy.std(uniquenessData.values()))

            #generate meanTokenFrequency and top300TokenFrequency, with stdDevs
            frequencyData = collections.Counter(title)

            self.meanTokenFrequency.append(numpy.mean(frequencyData.values()))
            self.stdDevTokenFrequency.append(numpy.std(frequencyData.values()))

            self.top300TokenFrequency.append(numpy.mean([x[1] for x in frequencyData.most_common(300)]))
            self.stdDevTop300TokenFrequency.append(numpy.std([x[1] for x in frequencyData.most_common(300)]))

            del tempList
            del uniquenessData
            del frequencyData

    def write(self):
        for index, title in enumerate(self.rowHeads):
            self.toWrite.append('')
            self.toWrite[index] += self.nGramParam + '-gram,'
            self.toWrite[index] += self.stemmerParam + ','
            self.toWrite[index] += title + ','
            self.toWrite[index] += str(self.textLength[index]) + ','
            self.toWrite[index] += str(self.lengthCoefficient[index]) + ','
            self.toWrite[index] += str(self.totalDistinctTokens[index]) + ','
            self.toWrite[index] += str(self.uniqueTokens[index]) + ','
            self.toWrite[index] += str(self.meanTokenUniqueness[index]) + ','
            self.toWrite[index] += str(self.stdDevTokenUniqueness[index]) + ','
            self.toWrite[index] += str(self.meanTokenFrequency[index]) + ','
            self.toWrite[index] += str(self.stdDevTokenFrequency[index]) + ','
            self.toWrite[index] += str(self.top300TokenFrequency[index]) + ','
            self.toWrite[index] += str(self.stdDevTop300TokenFrequency[index]) + '\n'

        print self.toWrite[0]

    def headerWrite(self):
        s = ',,,'
        for measures in self.firstRow:
            s += measures
        s += '\n'
        return s


def readFiles(path):
    wordLists = []
    for file in os.listdir(path):
        a = open(path + file, mode='rb').read()
        # if not POS, remove punctuation and convert to lowecase
        if not isPOS:
            a = nltk.word_tokenize(a.translate(None, string.punctuation).lower())
            if useStopList:
                a = [word for word in a if not word.decode('utf-8') in stopwords.words('english')]
                print 'x'
        wordLists.append(a)
    return wordLists

readTextPath = os.path.join('.', 'inputs', 'chs')
readPOSPath = os.path.join('.', 'inputs', 'chs-pos')
writePath = os.path.join('.', 'outputs')

fileNames = []
sourcePaths = []
chapterNames = ['01 Telemachus', '02 Nestor', '03 Proteus', '04 Calypso', '05 Lotus Eaters', '06 Hades', '07 Aeolus',
                '08 Laestrogynians', '09 Scylla and Charybdis', '10 Wandering Rocks', '11 Sirens', '12 Cyclops',
                '13 Nausicaa', '14 Oxen in the Sun', '15 Circe', '16 Eumaeus', '17 Ithaca', '18 Penelope']
stemmerModules = [['none', 'none'], ['lancaster', 'LancasterStemmer'], ['porter', 'PorterStemmer']]
NGRAMS = range(1, 11)
isPOS = False
useStopList = False
# open, word_tokenize(), and lower() texts
texts = readFiles(readTextPath)

outputName = time.strftime("%Y-%m-%d %H%M") + ' - '

outputName += 'POS Tokens - ' if isPOS else 'Word Tokens - '
outputName += 'Stopwords Removed.txt' if useStopList else 'Stopwords Included.txt'

stemmedTexts = []

#create stemmed lists
for i in stemmerModules:
    if not isPOS:
        if not (i[0] == 'none'):
            stemmer = getattr(getattr(nltk.stem, i[0]), i[1])()
            print stemmer
            stemmedTexts.append([[stemmer.stem(word.decode('utf-8')) for word in document] for document in texts])
        else:
            stemmedTexts.append(texts)
    else:
        stemmedTexts.append(texts)
        break

corpora = []
#create n-gram lists
for i in NGRAMS:
    for j in xrange(0, len(stemmedTexts)):
        print len(stemmedTexts[j])
        ngramTexts = []
        for k in stemmedTexts[j]:
            ngramTexts.append([a for a in ngrams(k, i)])  #test
        corpora.append(Corpus(ngramTexts, chapterNames, i, j))

for corpus in corpora:
    corpus.process()
    gc.collect()
    corpus.write()

p = os.path.join(writePath, time.strftime("%Y-%m-%d")+' data')

if not os.path.exists(p):
    os.makedirs(p)

f = open(p + outputName, 'w')
f.write("%s" % corpora[0].headerWrite())

print 'concatenating text'

for corpus in corpora:
    for line in corpus.toWrite:
        f.write("%s" % line)
f.close()
