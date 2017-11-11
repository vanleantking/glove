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

                        sentence = [c for c in sentence if not re.search(r'^/[/]?', c)]
                        sentence = [c for c in sentence if not re.search(r'_+', c)]
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


   
if __name__ == '__main__':
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


#     model = tf_glove.GloVeModel(embedding_size=50, context_size=1)
#     model.fit_to_corpus(corpus)
#     embedding = model.restore()
#     if len(embedding) > 0:
#         model.setembedding(embedding)
#     else:
#         model.train(num_epochs=150, log_dir="log/example", summary_batch_interval=1000)

#     num = 1
#     # devices = []
#     # dates = []
#     # ages = []
#     # phones = []
#     # emails = []
#     # Faxes = []
#     # urls = []
#     # devices = []
#     # ids = []
#     # zips = []
#     # usernames = []
#     # locationothers = []
#     professions = []
#     professionIndex = []
#     for patientRecord in onto.Patient.instances():
#         #data per patient
#         patients = []
#         doctors = []
#         doctorIndex = [] 
#         cities = []
#         streets = []        
#         hospitals = []        
#         countries = []
#         organizations = []        
#         states = []
        
#         num +=1
#         #patient name in each patients record
#         for patient in patientRecord.hasName:
#             # patients.append(model.embeded_phrases(patient))
#             patients.append(patient)

#         #get data for each medical record
#         for medicalRecord in patientRecord.was_recorded_at:
#             doctorRecord = []
#             professionRecord = []
#             # , professionRecord, citieRecord, streetRecord, stateRecord, hospitalRecord, organizationRecord, countrieRecord

#             doctorRecord = [model.embeded_phrases(doctor.hasName[0]) for doctor in medicalRecord.doctor_dianose]
#             doctorIndex.extend([{"name": doctor, "value": model.embeded_phrases(doctor.hasName[0])} for doctor in medicalRecord.doctor_dianose])
#             # citieRecord = [model.embeded_phrases(city.hasLocation) for city in medicalRecord.has_city]
#             # streetRecord = [model.embeded_phrases(street.hasLocation) for street in medicalRecord.has_street]
#             # stateRecord = [model.embeded_phrases(state.hasLocation) for state in medicalRecord.has_state]
#             # hospitalRecord = [model.embeded_phrases(hospital.hasLocation) for hospital in medicalRecord.has_hospital]
#             # organizationRecord = [model.embeded_phrases(organization.hasLocation) for organization in medicalRecord.has_organization]
#             # countrieRecord = [model.embeded_phrases(country.hasLocation) for country in medicalRecord.has_country]
#             # print(professionRecord)
#             doctors.extend(doctorRecord)
#             if len(medicalRecord.job_position) > 0:
#                 professionRecord = [model.embeded_phrases(profession.jobName[0]) for profession in medicalRecord.job_position]
#                 professionIndex.extend([{"name": profession, "value": model.embeded_phrases(profession.jobName[0])} for profession in medicalRecord.job_position])
#                 # print(medicalRecord.job_position)
#                 professions.extend(professionRecord)
#         # print(doctors)
#         # D = pairwise_distances(doctors, metric='cosine')        


#         # KMeans clustering
#         # print(doctorIndex)
#         # docs = BiKMeans()
#         # print("KMean clustering result---------------------------------------------")
#         # print(docs.execute(doctors))
#         # break
        
        
#         # print(docscluster)
#         # print(len(doctors))
#         Zdocs = linkage(doctors, 'single', 'cosine')
#         fig = plt.figure(figsize=(25, 10))
#         dn = dendrogram(Zdocs)

#         clustersdocs = fcluster(Zdocs, 0.55, criterion='distance')
#         clusters_number =  len(np.unique(clustersdocs))
#         # print(clusters_number)
#         docscluster = {i : [] for i in range(1, clusters_number+1)}
#         # for index, label in enumerate(clusters):
#         for index, docIndex in enumerate(doctorIndex):
#             if np.array_equal(docIndex['value'],doctors[index]):
#                 docscluster[clustersdocs[index]].append(docIndex['name'] )

#         docsclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecord.hasRecordName[0]))
        
#         # print(clustersdocs)
#         pcp = PostClusterProcessing()
#         for i in range(1, clusters_number+1):
#             for j in range (i+1, clusters_number+1):
#                 if i in docscluster.keys() and j in docscluster.keys():
#                     merged = pcp.mergecluster(docscluster[i], docscluster[j])
#                     if merged:
#                         try:
#                             docscluster[i].extend(docscluster[j])
#                             del docscluster[j]
#                         except:
#                             pass

#         docsclustering.write('\n')
#         for k, v in docscluster.items():
#             docsclustering.write('cluster ' + str(k))
#             docsclustering.write(''.join(str(v)))
#             docsclustering.write('\n')
#         docsclustering.write('\n\n\n\n\n')


#     # print(professionIndex)
#     Z = linkage(professions, 'single', 'cosine')
#     fig = plt.figure(figsize=(25, 10))
#     dn = dendrogram(Z)

#     clusters = fcluster(Z, 0.5, criterion='distance')
#     clusters_number =  len(np.unique(clusters))
#     # print(clusters)
#     docscluster = {i : [] for i in range(1, clusters_number+1)}
#     # for index, label in enumerate(clusters):
#     for index, docIndex in enumerate(professionIndex):
#         if np.array_equal(docIndex['value'],professions[index]):
#             docscluster[clusters[index]].append(docIndex['name'])

#     for k, v in docscluster.items():
#         print(k,v)
#     professioncluster.write('\n')
#     for k, v in docscluster.items():
#         professioncluster.write('cluster ' + str(k))
#         professioncluster.write(''.join(str(v)))
#         professioncluster.write('\n')
#     professioncluster.write('\n\n\n\n\n')   

#     docsclustering.close()
#     professioncluster.close()

#     # bikmeans = KMeans(n_clusters=2, random_state=0).fit(patitens)
#     # print(bikmeans.labels_)
    
#     # model.generate_tsne(path='log/tsne')


    

