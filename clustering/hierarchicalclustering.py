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
from nameprocessing import PostNameClusterProcessing, PreProcessingText, PostProfessionClusterProcessing
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corpus.tf_glove import GloVeModel
from corpus.read_data_tf import load_corpus, learning_phrase
from owlready2 import *


CORPUS_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/corpus'
RESULT_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/clustering/result/hierarchicalclustering/'
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


 
def cosine_simi(vector1, vector2):
    prod = 0.0
    
    mag1 = 0.0
    mag2 = 0.0
    
    for index, value in enumerate(vector1):
        prod += vector1[index] * vector2[index]
        mag1 += vector1[index] * vector1[index]
        mag2 += vector2[index] * vector2[index]
    
    return prod / (math.sqrt(mag1) * math.sqrt(mag2))

def hierarchical(datas, dataIndex, distance):
    Zdocs = linkage(datas, 'single', 'cosine')
        # fig = plt.figure(figsize=(25, 10))
        # dn = dendrogram(Zdocs)

    clustersdocs = fcluster(Zdocs, distance, criterion='distance')
    clusters_number =  len(np.unique(clustersdocs))

    docscluster = {i : [] for i in range(1, clusters_number+1)}

    for index, docIndex in enumerate(dataIndex):
        if np.array_equal(docIndex['value'],datas[index]):
            docscluster[clustersdocs[index]].append(docIndex['name'] )

    return docscluster, clusters_number

def logfile(file, data):
    file.write('\n')
    for k, v in data.items():
        file.write('cluster ' + str(k))
        file.write(''.join(str(v)))
        file.write('\n')
    file.write('\n\n\n\n\n')
    file.close()

def logdocsfile(file, data):
    file.write('\n')
    for k, v in data.items():
        file.write('cluster ' + str(k))
        file.write(''.join(str(v)))
        file.write('\n')
    file.write('\n\n\n\n\n')
    # file.close()
    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/newemr.owl").load()
    embedding = None
    docslogclustering = open(doctorclusterpath,"w")
    professionlogclustering = open(professionclusterpath,"w")
    citylogclustering = open(cityclusterpath,"w")
    statelogclustering = open(stateclusterpath,"w")
    streetlogclustering = open(streetclusterpath,"w")
    countrylogclustering = open(countryclusterpath,"w")
    hospitallogclustering = open(hospitalclusterpath,"w")
    organizationlogclustering = open(organizationclusterpath,"w")
    pre = PreProcessingText()

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
    cities = []
    cityIndex = []
    stateIndex = []
    states = []
    countries = []
    countryIndex = []
    streetIndex = []
    streets = []
    organizationIndex = []
    organizations = []
    hospitalIndex = []
    hospitals = []
    for patientRecord in onto.Patient.instances():
        #data per patient
        patients = []
        doctors = []
        doctorIndex = []       
        
        num +=1
        #patient name in each patients record
        for patient in patientRecord.hasName:
            # patients.append(model.embeded_phrases(patient))
            patients.append(patient)

        #get data for each medical record
        for medicalRecord in patientRecord.was_recorded_at:
            doctorRecord = []
            professionRecord = []
            cityRecord = []
            stateRecord = []
            countryRecord = []
            hospitalRecord = []
            streetRecord = []
            organizationRecord = []
            # , professionRecord, citieRecord, streetRecord, , hospitalRecord, organizationRecord, countrieRecord

            doctorRecord = [model.embeded_phrases(doctor.hasName[0]) for doctor in medicalRecord.doctor_dianose if np.isnan(model.embeded_phrases(doctor.hasName[0])).any() == False]
            doctorIndex.extend([{"name": doctor, "value": model.embeded_phrases(doctor.hasName[0])} for doctor in medicalRecord.doctor_dianose if np.isnan(model.embeded_phrases(doctor.hasName[0])).any() == False])
            # streetRecord = [model.embeded_phrases(street.hasLocation) for street in medicalRecord.has_street]
            # stateRecord = [model.embeded_phrases(state.hasLocation) for state in medicalRecord.has_state]
            # hospitalRecord = [model.embeded_phrases(hospital.hasLocation) for hospital in medicalRecord.has_hospital]
            # organizationRecord = [model.embeded_phrases(organization.hasLocation) for organization in medicalRecord.has_organization]
            # countrieRecord = [model.embeded_phrases(country.hasLocation) for country in medicalRecord.has_country]
            # print(professionRecord)
            doctors.extend(doctorRecord)
            if len(medicalRecord.job_position) > 0:
                professionRecord = [model.embeded_phrases(profession.jobName[0]) for profession in medicalRecord.job_position if np.isnan(model.embeded_phrases(profession.jobName[0])).any() == False]
                professionIndex.extend([{"name": profession, "value": model.embeded_phrases(profession.jobName[0])} for profession in medicalRecord.job_position if np.isnan(model.embeded_phrases(profession.jobName[0])).any() == False])
                professions.extend(professionRecord)

            if len(medicalRecord.has_city) > 0:
                cityRecord = [model.embeded_phrases(city.hasLocation[0]) for city in medicalRecord.has_city if np.isnan(model.embeded_phrases(city.hasLocation[0])).any() == False]
                cityIndex.extend([{"name": city, "value": model.embeded_phrases(city.hasLocation[0])} for city in medicalRecord.has_city if np.isnan(model.embeded_phrases(city.hasLocation[0])).any() == False])
                cities.extend(cityRecord)

            if len(medicalRecord.has_state) > 0:
                stateRecord = [model.embeded_phrases(state.name) for state in medicalRecord.has_state if np.isnan(model.embeded_phrases(state.name)).any() == False]
                stateIndex.extend([{"name": state, "value": model.embeded_phrases(state.name)} for state in medicalRecord.has_state if np.isnan(model.embeded_phrases(state.name)).any() == False])
                states.extend(stateRecord)

            if len(medicalRecord.has_country) > 0:
                countryRecord = [model.embeded_phrases(country.hasLocation[0]) for country in medicalRecord.has_country if np.isnan(model.embeded_phrases(country.hasLocation[0])).any() == False]
                countryIndex.extend([{"name": country, "value": model.embeded_phrases(country.hasLocation[0])} for country in medicalRecord.has_country if np.isnan(model.embeded_phrases(country.hasLocation[0])).any() == False])
                countries.extend(countryRecord)

            if len(medicalRecord.has_organization) > 0:
                organizationRecord = [model.embeded_phrases(organization.name) for organization in medicalRecord.has_organization if np.isnan(model.embeded_phrases(organization.name)).any() == False]
                organizationIndex.extend([{"name": organization, "value": model.embeded_phrases(organization.name)} for organization in medicalRecord.has_organization if np.isnan(model.embeded_phrases(organization.name)).any() == False])
                organizations.extend(organizationRecord)

            if len(medicalRecord.has_street) > 0:
                streetRecord = [model.embeded_phrases(street.hasLocation[0]) for street in medicalRecord.has_street if np.isnan(model.embeded_phrases(street.hasLocation[0])).any() == False]
                streetIndex.extend([{"name": street, "value": model.embeded_phrases(street.hasLocation[0])} for street in medicalRecord.has_street if np.isnan(model.embeded_phrases(street.hasLocation[0])).any() == False])
                streets.extend(streetRecord)

            if len(medicalRecord.record_from_hospital) > 0:
                hospitalRecord = [model.embeded_phrases(hospital.name) for hospital in medicalRecord.record_from_hospital if np.isnan(model.embeded_phrases(hospital.name)).any() == False]
                hospitalIndex.extend([{"name": hospital, "value": model.embeded_phrases(hospital.name)} for hospital in medicalRecord.record_from_hospital if np.isnan(model.embeded_phrases(hospital.name)).any() == False])
                hospitals.extend(hospitalRecord)

        docscluster, clusters_number = hierarchical(doctors, doctorIndex, 0.55)

        docslogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecord.hasRecordName[0]))
        
        #post processing for doctors
        pcp = PostNameClusterProcessing(pre)
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

        logdocsfile(docslogclustering, docscluster)


    professionscluster, clusters_number = hierarchical(professions, doctorIndex, 0.15)
    citiescluster, clusters_city_number = hierarchical(cities, cityIndex, 0.15)
    statescluster, clusters_city_number = hierarchical(states, stateIndex, 0.15)
    countriescluster, clusters_city_number = hierarchical(countries, countryIndex, 0.15)
    streetscluster, clusters_city_number = hierarchical(streets, streetIndex, 0.15)
    hospitalscluster, clusters_city_number = hierarchical(hospitals, hospitalIndex, 0.15)
    organizationscluster, clusters_city_number = hierarchical(organizations, organizationIndex, 0.15)
    # statescluster, clusters_city_number = hierarchical(states, cityIndex, 0.15)
    # statescluster, clusters_city_number = hierarchical(states, cityIndex, 0.15)
    
    #post processing for profession
    # ppcp = PostProfessionClusterProcessing(pre)
    # for i in range(1, clusters_number+1):
    #     for j in range (i+1, clusters_number+1):
    #         if i in docscluster.keys() and j in docscluster.keys():
    #             merged = ppcp.mergecluster(docscluster[i], docscluster[j])
    #             if merged:
    #                 try:
    #                     docscluster[i].extend(docscluster[j])
    #                     del docscluster[j]
    #                 except:
    #                     pass
    
    logfile(professionlogclustering, professionscluster)
    logfile(citylogclustering, citiescluster)
    logfile(statelogclustering, statescluster)
    logfile(countrylogclustering, countriescluster)
    logfile(streetlogclustering, streetscluster)
    logfile(hospitallogclustering, hospitalscluster)
    logfile(organizationlogclustering, organizationscluster)
    docslogclustering.close()
    

    print('log cluster success')

    # bikmeans = KMeans(n_clusters=2, random_state=0).fit(patitens)
    # print(bikmeans.labels_)
    
    # model.generate_tsne(path='log/tsne')


    

