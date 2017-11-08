import numpy as np
import tensorflow as tf
from collections import Counter, defaultdict
import tf_glove
import re
import os
import sys
import pickle
import bs4 as bs
import unicodedata
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.cluster.hierarchy import dendrogram, linkage
from bikmeans import BiKMeans
from KMedoids import KMedoids
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from nameprocessing import PostClusterProcessing
# import kmedoids
from owlready2 import *


BASE_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/data'
CORPUS_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/examples'
corpus_file = os.path.join(CORPUS_DIR_DATA, 'tf_corpus.p')


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

                        sentence = [c for c in sentence if not re.search(r'^/[/]?', c)]
                        #remove digits in sentence
                        sentence = [word for word in sentence if not re.search(r'\d+(\.\s?\d+)?\w*', word)]
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
    file = open("learning_phrase.txt","w") 
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
                    file.write(str(word_score) + " " + line[i] + " " + line[i+1] + " " + str(ngram_counter[line[i]+"_"+line[i+1]]) + " " + str(min_count) + " " + str(ngram_counter[line[i]]) + " " + str(ngram_counter[line[i+1]]) + " " + str(corpus_size))
                    file.write('\n')
                    text.insert(i+2, line[i]+"_"+line[i+1])
                    
                else:
                    del ngram_counter[line[i]+"_"+line[i+1]]
                i = i + 1
        corpuses.append(text)
    file.close()
    corpus_dict['corpus'] = corpuses
    corpus_dict['corpus_size'] = corpus_size
    pickle.dump( corpus_dict, open( "tf_corpus.p", "wb" ) )
    return corpuses


def load_data():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    texts = []  # list of text samples
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
                lines = re.split(r'[\s\.][\.\n]+', data)
                for line in lines:
                    if line.strip():
                        yield nltk.wordpunct_tokenize(line.strip())
    
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/newemr.owl").load()
    embedding = None
    docsclustering = open("doctorcluster.txt","w")
    if os.path.isfile(corpus_file):
        corpus_dict = pickle.load( open( corpus_file, "rb" ) )
        corpus = corpus_dict['corpus']
        corpus_size = corpus_dict['corpus_size']
    else:
        corpus, corpus_size = load_corpus()
        corpus = learning_phrase(corpus, corpus_size)
    print('learning phrase completed')
    file = open("corpus.txt","w")
    for text in corpus:
        file.write(' '.join(text))
        file.write('\n')
    file.close()

    print('log corpus completed')


    model = tf_glove.GloVeModel(embedding_size=100, context_size=1)
    model.fit_to_corpus(corpus)
    embedding = model.restore()
    if len(embedding) > 0:
        model.setembedding(embedding)
    else:
        model.train(num_epochs=150, log_dir="log/example", summary_batch_interval=1000)

    num = 1
    # devices = []
    # dates = []
    # ages = []
    # phones = []
    # emails = []
    # Faxes = []
    # urls = []
    # devices = []
    # ids = []
    # zips = []
    # usernames = []
    # locationothers = []
    for patientRecord in onto.Patient.instances():
        #data per patient
        patients = []
        doctors = []
        doctorIndex = [] 
        cities = []
        streets = []        
        hospitals = []        
        countries = []
        organizations = []        
        professions = []        
        states = []
        
        num +=1
        #patient name in each patients record
        for patient in patientRecord.hasName:
            # patients.append(model.embeded_phrases(patient))
            patients.append(patient)

        #get data for each medical record
        for medicalRecord in patientRecord.was_recorded_at:
            doctorRecord = []
            # , professionRecord, citieRecord, streetRecord, stateRecord, hospitalRecord, organizationRecord, countrieRecord 
            doctorRecord = [doctor.hasName for doctor in medicalRecord.doctor_dianose]
            doctorRecord = [model.embeded_phrases(doctor.hasName[0]) for doctor in medicalRecord.doctor_dianose]
            doctorIndex.extend([{"name": doctor, "value": model.embeded_phrases(doctor.hasName[0])} for doctor in medicalRecord.doctor_dianose])
            # professionRecord = [model.embeded_phrases(profession.jobName) for profession in medicalRecord.job_position]
            # citieRecord = [model.embeded_phrases(city.hasLocation) for city in medicalRecord.has_city]
            # streetRecord = [model.embeded_phrases(street.hasLocation) for street in medicalRecord.has_street]
            # stateRecord = [model.embeded_phrases(state.hasLocation) for state in medicalRecord.has_state]
            # hospitalRecord = [model.embeded_phrases(hospital.hasLocation) for hospital in medicalRecord.has_hospital]
            # organizationRecord = [model.embeded_phrases(organization.hasLocation) for organization in medicalRecord.has_organization]
            # countrieRecord = [model.embeded_phrases(country.hasLocation) for country in medicalRecord.has_country]

            doctors.extend(doctorRecord)
        # print(doctors)
        # D = pairwise_distances(doctors, metric='cosine')        


        #KMeans clustering
        # docs = KMeans(n_clusters=4, random_state=0).fit(doctors)
        # print("KMean clustering result---------------------------------------------")
        # print(docs.labels_)
        
        
        # print(docscluster)
        # print(len(doctors))
        try:
            Z = linkage(doctors, 'single', 'cosine')
        except:
            print(num, doctors)
            print(doctorIndex)
            # return
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        c, coph_dists = cophenet(Z, pdist(doctors, metric='cosine'))
        # print(c)
        # print(Z)
        clusters = fcluster(Z, 0.5, criterion='distance')
        clusters_number =  len(np.unique(clusters))
        print(clusters_number)
        docscluster = {i : [] for i in range(1, clusters_number+1)}
        # for index, label in enumerate(clusters):
        for index, docIndex in enumerate(doctorIndex):
            if np.array_equal(docIndex['value'],doctors[index]):
                docscluster[clusters[index]].append(docIndex['name'] )

        docsclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecord.hasRecordName[0]))
        
        # print(clusters)
        pcp = PostClusterProcessing()
        for i in range(1, clusters_number+1):
            for j in range (i+1, clusters_number+1):
                if i in docscluster.keys() and j in docscluster.keys():
                    merged = pcp.mergecluster(docscluster[i], docscluster[j])
                    if merged:
                        try:
                            docscluster[i].extend(docscluster[j])
                            del docscluster[j]
                        except:
                            pass

        docsclustering.write('\n')
        for k, v in docscluster.items():
            docsclustering.write('cluster ' + str(k))
            docsclustering.write(''.join(str(v)))
            docsclustering.write('\n')
        docsclustering.write('\n\n\n\n\n')
        # for doc in docscluster:
        #     print(doc.values())

        # print(fancy_dendrogram(
        #     Z,
        #     truncate_mode='lastp',
        #     p=12,
        #     leaf_rotation=90.,
        #     leaf_font_size=12.,
        #     show_contracted=True,
        #     annotate_above=10,
        #     max_d=0.5,  # plot a horizontal cut-off line
        # ))
        # plt.show()
        # D = 1 - cdist(doctors, doctors, 'cosine')
        # print(D, doctors)
        # M, C = kmedoids.kMedoids(D, 4)
        # print('')
        # print('clustering result:')
        # for label in C:
        #     for point_idx in C[label]:
        #         print('label {0}:　{1}'.format(label, doctors[point_idx]))
        # docscluster = {}

        # for index, label in enumerate(docs.labels_):
        #     docscluster[label].append(doctors[index])
        #     states = 
        #     medicalRecord.was_use_device
        # if (num == 10):
        #     break

    docsclustering.close()
    # patitens = []
    # for i in onto.Patient.instances():
    #     patitens.append(model.embeded_phrases(i.name))

    # bikmeans = KMeans(n_clusters=2, random_state=0).fit(patitens)
    # print(bikmeans.labels_)

    # doctors = []
    # l_doctors = []
    # for i in onto.Doctor.instances():
    #     doctors.append(i.name)
    #     l_doctors.append(model.embeded_phrases(i.name))

    # docs = KMeans(n_clusters=10, random_state=0).fit(l_doctors)
    # print(cosine_similarity([model.embeded_phrases("Yechiel Kidd")], [model2.embeded_phrases("Kidd")]))
    # print(cosine_similarity([model.embeded_phrases("Yechiel Kidd")], [model.embeded_phrases("Kidd")]))
    # print(cosine_similarity([model.embeded_phrases("Y Ullrich")], [model.embeded_phrases("Y. Ullrich")]))
    # print(cosine_similarity([model.embeded_phrases("Y Ullrich")], [model.embeded_phrases("Ullrich")]))
    # print(cosine_similarity([model.embeded_phrases("Yosef Q Ullrich")], [model.embeded_phrases("Ullrich")]))
    # print(cosine_similarity([model.embeded_phrases("Y Ullrich")], [model.embeded_phrases("Y. Ullrich")]))
    # print(cosine_similarity([model.embeded_phrases("Ullysses B. Gilbert")], [model.embeded_phrases("Ullysses Gilbert")]))
    # print(cosine_similarity([model.embeded_phrases("Ullysses B. Gilbert")], [model.embeded_phrases("Ullysses Gilbert")]))
    # print(cosine_similarity([model.embeded_phrases("Ullysses B. Gilbert")], [model.embeded_phrases("Gilbert")]))
    # print(cosine_similarity([model.embeded_phrases("Gilbert")], [model.embeded_phrases("Ullysses Gilbert")]))
    # print(cosine_similarity([model.embeded_phrases("Yosef Q Ullrich")], [model.embeded_phrases("Herman N. Weller")]))
    # print(cosine_similarity(l_doctors))
    # print(model.embeded_phrases("Betty Kaitlin Wood"))

    # print(doctors)
    # print(docs.labels_)
    
    # model.generate_tsne(path='log/tsne')


    

