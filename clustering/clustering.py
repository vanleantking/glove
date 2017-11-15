import numpy as np
import tensorflow as tf
from collections import Counter, defaultdict
import re
import math
import os
import sys
import pickle
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster, fclusterdata
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from nameprocessing import PreProcessingText, PostProcessing, PostHospitalClusterProcessing, PostNameClusterProcessing
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corpus.tf_glove import GloVeModel
from corpus.read_data_tf import load_corpus, learning_phrase


CORPUS_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/corpus'
corpus_log = os.path.join(CORPUS_DIR_DATA, 'corpus.txt')
corpus_file = os.path.join(CORPUS_DIR_DATA, 'tf_corpus.p')
class Clustering:
    
    
    def __init__(self):
        self.model = self.setGloveModel()
        self.pre = PreProcessingText()
  
    def get_model(self):
        return self.model
      
    def get_pre(self):
        return self.pre

    def setGloveModel(self, is_abbreviation = False):
        if os.path.isfile(corpus_file):
            corpus_dict = pickle.load( open( corpus_file, "rb" ) )
            corpus = corpus_dict['corpus']
            corpus_size = corpus_dict['corpus_size']
        else:
            corpus, corpus_size = load_corpus()
            corpus = learning_phrase(corpus, corpus_size)
        print('learning phrase completed')
        file = open(corpus_log,"w")
        for text in corpus:
            file.write(' '.join(text))
            file.write('\n')
        file.close()

        print('log corpus completed')


        model = GloVeModel(embedding_size=50, context_size=1)
        model.fit_to_corpus(corpus)
        embedding = model.restore()
        if len(embedding) > 0:
            model.setembedding(embedding)
        else:
            model.train(num_epochs=150, log_dir="log/example", summary_batch_interval=1000)

        return model


    def getword2vectabb(self, obj, max_length, is_abb = False):
        if is_abb:
            return np.concatenate((np.array(self.get_model().embeded_phrases(obj)), np.asarray(self.getabbreviation(obj, max_length), dtype=np.float32)), axis=0)
        
        return self.get_model().embeded_phrases(obj)

    def getmaxlengthabb(self, datas, is_username = False):
        result = []
        for data in datas:
            abb, is_abb = self.pre.abbreviation(data, is_username)
            result.append(len(abb))
        return max(result)

    def getabbreviation(self, obj, max_length, is_username= False):
        abb, is_abb = self.pre.abbreviation(obj, is_username)
        length_abb = len(abb)
        result = []
        for index in range(max_length):
            if index < length_abb:
                result.append(ord(abb[index]))
            else:
                result.append(0)
        return result


    def logfile(self, file, data):
        file.write('\n')
        for k, v in data.items():
            file.write('cluster ' + str(k))
            file.write(''.join(str(v)))
            file.write('\n')
        file.write('\n\n\n\n\n')
        file.close()

    def logdocsfile(self, file, data):
        file.write('\n')
        for k, v in data.items():
            file.write('cluster ' + str(k))
            file.write(''.join(str(v)))
            file.write('\n')
        file.write('\n\n\n\n\n')