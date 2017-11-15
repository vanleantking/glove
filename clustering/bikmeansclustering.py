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
from owlready2 import *


CORPUS_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/corpus'
RESULT_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/clustering/result/bikmeansclustering/'
CURRENT_FOLDER = os.path.dirname(os.path.realpath('__file__'))
corpus_log = os.path.join(CORPUS_DIR_DATA, 'corpus.txt')
corpus_file = os.path.join(CORPUS_DIR_DATA, 'tf_corpus.p')
doctorclusterpath = os.path.join(RESULT_DIR_DATA, 'doctorcluster.txt')
patientclusterpath = os.path.join(RESULT_DIR_DATA, 'patientcluster.txt')
professionclusterpath = os.path.join(RESULT_DIR_DATA, 'professioncluster.txt')

#location clustering log path
cityclusterpath = os.path.join(RESULT_DIR_DATA, 'citycluster.txt')
stateclusterpath = os.path.join(RESULT_DIR_DATA, 'statecluster.txt')
streetclusterpath = os.path.join(RESULT_DIR_DATA, 'streetcluster.txt')
organizationclusterpath = os.path.join(RESULT_DIR_DATA, 'organizationcluster.txt')
hospitalclusterpath = os.path.join(RESULT_DIR_DATA, 'hospitalcluster.txt')
countryclusterpath = os.path.join(RESULT_DIR_DATA, 'countrycluster.txt')
usernameclusterpath = os.path.join(RESULT_DIR_DATA, 'usernamecluster.txt')

class BiKmeanClustering:
	

