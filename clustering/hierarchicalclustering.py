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
from nameprocessing import PostClusterProcessing
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corpus.tf_glove import GloVeModel
from corpus.read_data_tf import load_corpus, learning_phrase
from owlready2 import *


CORPUS_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/corpus'
RESULT_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/result/hierarchicalclustering/'
doctorclusterpath = RESULT_DIR_DATA + 'doctorcluster.txt'
professionclusterpath = RESULT_DIR_DATA + 'professioncluster.txt'
corpus_file = os.path.join(CORPUS_DIR_DATA, 'tf_corpus.p')
corpus_log = os.path.join(CORPUS_DIR_DATA, 'corpus.txt')


 
def cosine_simi(vector1, vector2):
    prod = 0.0
    
    mag1 = 0.0
    mag2 = 0.0
    
    for index, value in enumerate(vector1):
        prod += vector1[index] * vector2[index]
        mag1 += vector1[index] * vector1[index]
        mag2 += vector2[index] * vector2[index]
    
    return prod / (math.sqrt(mag1) * math.sqrt(mag2))

    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/newemr.owl").load()
    embedding = None
    docsclustering = open('doctorcluster.txt',"w")
    professioncluster = open('professioncluster.txt',"w")
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
    professions = []
    professionIndex = []
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
        states = []
        
        num +=1
        #patient name in each patients record
        for patient in patientRecord.hasName:
            # patients.append(model.embeded_phrases(patient))
            patients.append(patient)

        #get data for each medical record
        for medicalRecord in patientRecord.was_recorded_at:
            doctorRecord = []
            professionRecord = []
            # , professionRecord, citieRecord, streetRecord, stateRecord, hospitalRecord, organizationRecord, countrieRecord

            doctorRecord = [model.embeded_phrases(doctor.hasName[0]) for doctor in medicalRecord.doctor_dianose]
            doctorIndex.extend([{"name": doctor, "value": model.embeded_phrases(doctor.hasName[0])} for doctor in medicalRecord.doctor_dianose])
            # citieRecord = [model.embeded_phrases(city.hasLocation) for city in medicalRecord.has_city]
            # streetRecord = [model.embeded_phrases(street.hasLocation) for street in medicalRecord.has_street]
            # stateRecord = [model.embeded_phrases(state.hasLocation) for state in medicalRecord.has_state]
            # hospitalRecord = [model.embeded_phrases(hospital.hasLocation) for hospital in medicalRecord.has_hospital]
            # organizationRecord = [model.embeded_phrases(organization.hasLocation) for organization in medicalRecord.has_organization]
            # countrieRecord = [model.embeded_phrases(country.hasLocation) for country in medicalRecord.has_country]
            # print(professionRecord)
            doctors.extend(doctorRecord)
            if len(medicalRecord.job_position) > 0:
                professionRecord = [model.embeded_phrases(profession.jobName[0]) for profession in medicalRecord.job_position]
                professionIndex.extend([{"name": profession, "value": model.embeded_phrases(profession.jobName[0])} for profession in medicalRecord.job_position])
                # print(medicalRecord.job_position)
                professions.extend(professionRecord)
        # print(doctors)
        # D = pairwise_distances(doctors, metric='cosine')        


        # KMeans clustering
        # print(doctorIndex)
        # docs = BiKMeans()
        # print("KMean clustering result---------------------------------------------")
        # print(docs.execute(doctors))
        # break
        
        
        # print(docscluster)
        # print(len(doctors))
        Zdocs = linkage(doctors, 'single', 'cosine')
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Zdocs)

        clustersdocs = fcluster(Zdocs, 0.55, criterion='distance')
        clusters_number =  len(np.unique(clustersdocs))
        # print(clusters_number)
        docscluster = {i : [] for i in range(1, clusters_number+1)}
        # for index, label in enumerate(clusters):
        for index, docIndex in enumerate(doctorIndex):
            if np.array_equal(docIndex['value'],doctors[index]):
                docscluster[clustersdocs[index]].append(docIndex['name'] )

        docsclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecord.hasRecordName[0]))
        
        # print(clustersdocs)
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


    # print(professionIndex)
    Z = linkage(professions, 'single', 'cosine')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)

    clusters = fcluster(Z, 0.5, criterion='distance')
    clusters_number =  len(np.unique(clusters))
    # print(clusters)
    docscluster = {i : [] for i in range(1, clusters_number+1)}
    # for index, label in enumerate(clusters):
    for index, docIndex in enumerate(professionIndex):
        if np.array_equal(docIndex['value'],professions[index]):
            docscluster[clusters[index]].append(docIndex['name'])

    for k, v in docscluster.items():
        print(k,v)
    professioncluster.write('\n')
    for k, v in docscluster.items():
        professioncluster.write('cluster ' + str(k))
        professioncluster.write(''.join(str(v)))
        professioncluster.write('\n')
    professioncluster.write('\n\n\n\n\n')   

    docsclustering.close()
    professioncluster.close()

    # bikmeans = KMeans(n_clusters=2, random_state=0).fit(patitens)
    # print(bikmeans.labels_)
    
    # model.generate_tsne(path='log/tsne')


    

