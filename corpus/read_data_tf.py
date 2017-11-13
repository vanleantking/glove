import numpy as np
import tensorflow as tf
from corpus import tf_glove
import re
import math
import os
import sys
import pickle
import bs4 as bs
import unicodedata
import nltk
from collections import Counter, defaultdict
from string import punctuation
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


BASE_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/data'
CORPUS_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/corpus'
corpus_file = os.path.join(CORPUS_DIR_DATA, 'tf_corpus.p')
corpus_log = os.path.join(CORPUS_DIR_DATA, 'corpus.txt')


def load_corpus():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    texts = []  # list of text samples
    doc = []
    words_train = 0
    stemmer = SnowballStemmer("english")
    for name in sorted(os.listdir(BASE_DIR_DATA)):
        path = os.path.join(BASE_DIR_DATA, name)
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='UTF-8')
                t = f.read()
                text = bs.BeautifulSoup(t, 'xml')
                data = text.TEXT.string
                tmp = unicodedata.normalize('NFKD', data).encode('ascii','ignore')
                # print(tmp)
                sents = sent_tokenize(tmp.decode("ascii"))
                for sent in sents:
                    if sent.strip():                        
                        
                        words_arrays = re.findall(r"[\w']+|[.,!?;\/+]", sent.strip().lower())

                        #remove date in sentence
                        sentence = [word for word in words_arrays if not re.search(r'[0-9]{1,4}[\/,:,-][0-9]{1,3}([\/,:,-][0-9]{2,4})?([\/,:,-][0-9]{2,4})?', word)]

                        # sentence = [word for word in sentence if not re.search(r'\'s', word)]

                        #remove specials symbols
                        sentence = [c for c in sentence if c not in ('!','.',':', '-', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%', '{', '}', '=', ',', ']', '[', '`', '\'')]

                        #remove // in text
                        sentence = [c for c in sentence if not re.search(r'^/[/]?', c)]

                        #remove ________ in text
                        sentence = [c for c in sentence if not re.search(r'_+', c)]

                        #split string 123tert to 123 tert
                        sentence = [c for word in sentence for c in re.split(r'([0-9]*)([a-zA-Z\'0-9]+)', word) if c]
                        #remove measure weight/digits in sentence
                        sentence = [word for word in sentence if not re.search(r'^\d+\.*\s?\d*\s?[mg]*$', word)]
                        sentence = [c for c in sentence if not re.search(r'^mg', c)]
                        

                        sentence = [c for c in sentence if not re.search(r'``|\'\'', c)]

                        #remove stopwords in sentence
                        # process_words = [word for word in sentence if word not in stopwords.words('english')]

                        #stemming words in sentences
                        # process_words = [stemmer.stem(t) for t in process_words]

                        #tokenizer sentence
                        words_train += len(sentence)
                        texts.append(sentence)
    return texts, words_train

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def construct_ngrams(corpus, n):
    ngrams = Counter()
    for line in corpus:
        ngram = ['_'.join(x) for x in find_ngrams(line, n)]
        ngrams.update(ngram)
    return ngrams


def learning_phrase(corpus, corpus_size, word_phrases = 4, min_count = 2, threshold = 100):
    phrase = 1
    ngram_counter = Counter()
    corpuses = []
    corpus_dict = {}
    while (phrase <= word_phrases):
        ngram_counter += construct_ngrams(corpus, phrase)
        phrase += 1
    for line in corpus:
        text = line
        i = 0
        while (i < len(line) - 1):
            if (ngram_counter[line[i]]) < min_count or (ngram_counter[line[i+1]]) < min_count :
                i += 1
            else:
                word_score = (ngram_counter[line[i]+"_"+line[i+1]] - min_count) / float(ngram_counter[line[i]]) / float(ngram_counter[line[i+1]]) * corpus_size;
                
                if word_score > threshold:
                    text.insert(i+2, line[i]+"_"+line[i+1])
                    
                else:
                    del ngram_counter[line[i]+"_"+line[i+1]]
                i = i + 1
        corpuses.append(text)
    corpus_dict['corpus'] = corpuses
    corpus_dict['corpus_size'] = corpus_size
    pickle.dump( corpus_dict, open( corpus_file, "wb" ) )
    return corpuses

