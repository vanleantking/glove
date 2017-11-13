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
usernameclusterpath = os.path.join(RESULT_DIR_DATA, 'usernamecluster.txt')

class HierachicalClustering:

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


    def cosine_simi(self, vector1, vector2):
        prod = 0.0
        
        mag1 = 0.0
        mag2 = 0.0
        
        for index, value in enumerate(vector1):
            prod += vector1[index] * vector2[index]
            mag1 += vector1[index] * vector1[index]
            mag2 += vector2[index] * vector2[index]
        
        return prod / (math.sqrt(mag1) * math.sqrt(mag2))

    def hierarchical(self, datas, dataIndex, distance):
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


    def getword2vectabb(self, obj, max_length, is_abb = False):
        if is_abb:
            return self.get_model().embeded_phrases(obj)
        else:
            return np.concatenate((np.array(self.get_model().embeded_phrases(obj)), np.asarray(self.getabbreviation(self, obj, max_length, is_username), dtype=np.float32)), axis=0)

    def getmaxlengthabb(self, datas, is_username = False):
        return max([(len(self.pre.abbreviation(data, is_username))) for data in datas])

    def getabbreviation(self, obj, max_length, is_username= False):
        abb, is_abb = self.pre.abbreviation(obj, is_username)
        length_abb = len(abb)
        result = []
        for index in range(max_length):
            if index < length_abb:
                result[index] = ord(abb[index])
            else:
                result[index] = 0
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



def constructioncluster(hc, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv = False):
    docslogclustering = open(doctorclusterpath,"w")
    professionlogclustering = open(professionclusterpath,"w")
    citylogclustering = open(cityclusterpath,"w")
    statelogclustering = open(stateclusterpath,"w")
    streetlogclustering = open(streetclusterpath,"w")
    countrylogclustering = open(countryclusterpath,"w")
    hospitallogclustering = open(hospitalclusterpath,"w")
    organizationlogclustering = open(organizationclusterpath,"w")
    usernamelogclustering = open(usernameclusterpath,"w")

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
    usernames = []
    usernameIndex = []
    for patientRecord in onto.Patient.instances():
        #data per patient
        patients = []
        doctors = []
        doctorIndex = []       
        
        #patient name in each patients record
        for patient in patientRecord.hasName:
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
            usernameRecord = []

            doctorRecord = [hc.getword2vectabb(doctor.hasName[0], doctorsmaxlength, is_abbrv) for doctor in medicalRecord.doctor_dianose if np.isnan(hc.get_model().embeded_phrases(doctor.hasName[0])).any() == False]
            doctorIndex.extend([{"name": doctor, "value": hc.getword2vectabb(doctor.hasName[0], doctorsmaxlength, is_abbrv)} for doctor in medicalRecord.doctor_dianose if np.isnan(hc.get_model().embeded_phrases(doctor.hasName[0])).any() == False])
            doctors.extend(doctorRecord)
            if len(medicalRecord.job_position) > 0:
                professionRecord = [hc.getword2vectabb(profession.jobName[0], professionsmaxlength, is_abbrv) for profession in medicalRecord.job_position if np.isnan(hc.get_model().embeded_phrases(profession.jobName[0])).any() == False]
                professionIndex.extend([{"name": profession, "value": hc.getword2vectabb(profession.jobName[0], professionsmaxlength, is_abbrv)} for profession in medicalRecord.job_position if np.isnan(hc.get_model().embeded_phrases(profession.jobName[0])).any() == False])
                professions.extend(professionRecord)

            if len(medicalRecord.has_city) > 0:
                cityRecord = [hc.getword2vectabb(city.hasLocation[0], citysmaxlength, is_abbrv) for city in medicalRecord.has_city if np.isnan(hc.get_model().embeded_phrases(city.hasLocation[0])).any() == False]
                cityIndex.extend([{"name": city, "value": hc.getword2vectabb(city.hasLocation[0], citysmaxlength, is_abbrv)} for city in medicalRecord.has_city if np.isnan(hc.get_model().embeded_phrases(city.hasLocation[0])).any() == False])
                cities.extend(cityRecord)

            if len(medicalRecord.has_state) > 0:
                stateRecord = [hc.getword2vectabb(state.name, statesmaxlength, is_abbrv) for state in medicalRecord.has_state if np.isnan(hc.get_model().embeded_phrases(state.name)).any() == False]
                stateIndex.extend([{"name": state, "value": hc.getword2vectabb(state.name, statesmaxlength, is_abbrv)} for state in medicalRecord.has_state if np.isnan(hc.get_model().embeded_phrases(state.name)).any() == False])
                states.extend(stateRecord)

            if len(medicalRecord.has_country) > 0:
                countryRecord = [hc.getword2vectabb(country.hasLocation[0], countrysmaxlength, is_abbrv) for country in medicalRecord.has_country if np.isnan(hc.get_model().embeded_phrases(country.hasLocation[0])).any() == False]
                countryIndex.extend([{"name": country, "value": hc.getword2vectabb(country.hasLocation[0], countrysmaxlength, is_abbrv)} for country in medicalRecord.has_country if np.isnan(hc.get_model().embeded_phrases(country.hasLocation[0])).any() == False])
                countries.extend(countryRecord)

            if len(medicalRecord.has_organization) > 0:
                organizationRecord = [hc.getword2vectabb(organization.name, organizationsmaxlength, is_abbrv) for organization in medicalRecord.has_organization if np.isnan(hc.get_model().embeded_phrases(organization.name)).any() == False]
                organizationIndex.extend([{"name": organization, "value": hc.getword2vectabb(organization.name, organizationsmaxlength, is_abbrv)} for organization in medicalRecord.has_organization if np.isnan(hc.get_model().embeded_phrases(organization.name)).any() == False])
                organizations.extend(organizationRecord)

            if len(medicalRecord.has_street) > 0:
                streetRecord = [hc.getword2vectabb(street.hasLocation[0], streetsmaxlength, is_abbrv) for street in medicalRecord.has_street if np.isnan(hc.get_model().embeded_phrases(street.hasLocation[0])).any() == False]
                streetIndex.extend([{"name": street, "value": hc.getword2vectabb(street.hasLocation[0], streetsmaxlength, is_abbrv)} for street in medicalRecord.has_street if np.isnan(hc.get_model().embeded_phrases(street.hasLocation[0])).any() == False])
                streets.extend(streetRecord)

            if len(medicalRecord.record_from_hospital) > 0:
                hospitalRecord = [hc.getword2vectabb(hospital.name, hospitalsmaxlength, is_abbrv) for hospital in medicalRecord.record_from_hospital if np.isnan(hc.get_model().embeded_phrases(hospital.name)).any() == False]
                hospitalIndex.extend([{"name": hospital, "value": hc.getword2vectabb(hospital.name, hospitalsmaxlength, is_abbrv)} for hospital in medicalRecord.record_from_hospital if np.isnan(hc.get_model().embeded_phrases(hospital.name)).any() == False])
                hospitals.extend(hospitalRecord)

            if len(medicalRecord.has_username) > 0:
                usernameRecord = [hc.getword2vectabb(username.hasName[0], usernamesmaxlength, is_abbrv) for username in medicalRecord.has_username if np.isnan(hc.get_model().embeded_phrases(username.hasName[0])).any() == False]
                usernameIndex.extend([{"name": username, "value": hc.getword2vectabb(username.hasName[0], usernamesmaxlength, is_abbrv)} for username in medicalRecord.has_username if np.isnan(hc.get_model().embeded_phrases(username.hasName[0])).any() == False])
                usernames.extend(usernameRecord)

        docscluster, clusters_number = hc.hierarchical(doctors, doctorIndex, 0.55)

        docslogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecord.hasRecordName[0]))
        
        #post processing for doctors
        pcp = PostNameClusterProcessing(hc.get_pre())
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

        hc.logdocsfile(docslogclustering, docscluster)

    professionscluster, clusters_number = hc.hierarchical(professions, professionIndex, 0.15)
    citiescluster, clusters_city_number = hc.hierarchical(cities, cityIndex, 0.15)
    statescluster, clusters_city_number = hc.hierarchical(states, stateIndex, 0.15)
    countriescluster, clusters_city_number = hc.hierarchical(countries, countryIndex, 0.15)
    streetscluster, clusters_city_number = hc.hierarchical(streets, streetIndex, 0.15)
    hospitalscluster, clusters_city_number = hc.hierarchical(hospitals, hospitalIndex, 0.45)
    organizationscluster, clusters_city_number = hc.hierarchical(organizations, organizationIndex, 0.15)
    usernamescluster, clusters_username_number = hc.hierarchical(usernames, usernameIndex, 0.15)
    
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
    
    hc.logfile(professionlogclustering, professionscluster)
    hc.logfile(citylogclustering, citiescluster)
    hc.logfile(statelogclustering, statescluster)
    hc.logfile(countrylogclustering, countriescluster)
    hc.logfile(streetlogclustering, streetscluster)
    hc.logfile(hospitallogclustering, hospitalscluster)
    hc.logfile(organizationlogclustering, organizationscluster)
    hc.logfile(usernamelogclustering, usernamescluster)
    docslogclustering.close()
    

    print('log cluster success')
    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/newemr.owl").load()
    print(onto.Doctor)
    
    for doctor in onto.Doctor.instances():
        print(doctor.name)
        break

    doctorsName = [doctor.name for doctor in onto.Doctor.instances()]
    professionsName = [profession.name for profession in onto.Profession.instances()]
    citysName = [city.name for city in onto.City.instances()]
    statesName = [state.name for state in onto.State.instances()]
    streetsName = [street.name for street in onto.Street.instances()]
    countrysName = [country.name for country in onto.Country.instances()]
    hospitalsName = [hospital.name for hospital in onto.Hospital.instances()]
    organizationsName = [organization.name for organization in onto.Organization.instances()]
    usernamesName = [username.name for username in onto.Username.instances()]

    hc = HierachicalClustering()

    doctorsmaxlength = hc.getmaxlengthabb(doctorsName, is_username = False)
    professionsmaxlength = hc.getmaxlengthabb(professionsName, is_username = False)
    citysmaxlength = hc.getmaxlengthabb(citysName, is_username = False)
    statesmaxlength = hc.getmaxlengthabb(statesName, is_username = False)
    streetsmaxlength = hc.getmaxlengthabb(streetsName, is_username = False)
    countrysmaxlength = hc.getmaxlengthabb(countrysName, is_username = False)
    hospitalsmaxlength = hc.getmaxlengthabb(hospitalsName, is_username = False)
    organizationsmaxlength = hc.getmaxlengthabb(organizationsName, is_username = False)
    usernamesmaxlength = hc.getmaxlengthabb(usernamesName, is_username = False)


    constructioncluster(hc, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv=True)

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
    # locationothers = []
    
    
    # model.generate_tsne(path='log/tsne')


    

