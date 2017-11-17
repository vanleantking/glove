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
from clustering import Clustering
from bikmeans import BiKMeans
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

class BiKmeanClustering(Clustering):

    def __init__(self):
        super().__init__()

    def get_cluster(self, X, dataIndex):
        bikmeans = BiKMeans()
        clusters = bikmeans.execute(X)
        docscluster = {i : [] for i in range(len(clusters))}

        for index, docIndex in enumerate(clusters):
            for doc in docIndex['vectors']:
                for k, v in enumerate(dataIndex):
                    if np.array_equal(v['value'],doc):
                        docscluster[index].append(v['name'] )

        return docscluster

def constructionbikmeancluster(hc, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv = False):
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
    usernames = []
    usernameIndex = []
    is_abbrv = False
    bi = BiKmeanClustering()
    for patientRecord in onto.Patient.instances():
        #data per patient
        patients = []
        doctors = []
        doctorIndex = []       
        hospitalIndex = []
        hospitals = []
        
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

        docsluster = bi.get_cluster(doctors, doctorIndex)
        # print(docsluster)
        # break
        
        
        # #post processing for doctors
        docslogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecord.hasRecordName[0]))
        # pncp = PostNameClusterProcessing(hc.get_pre())
        # docscluster = hc.postclustering(docscluster, clusters_number, pncp)
        hc.logdocsfile(docslogclustering, docsluster)

        if len(hospitals) > 1:
            hospitalscluster = bi.get_cluster(hospitals, hospitalIndex)
            # phcp = PostHospitalClusterProcessing(hc.get_pre())
            # hospitalscluster = hc.postclustering(hospitalscluster, clusters_number, phcp)
        else:
            hospitalscluster = {1 : []}
            hospitalscluster[1].append(hospitalIndex[0]['name'])
        print(hospitalscluster)
        hc.logdocsfile(hospitallogclustering, hospitalscluster)

    professionscluster = bi.get_cluster(professions, professionIndex)
    citiescluster = bi.get_cluster(cities, cityIndex)
    statescluster = bi.get_cluster(states, stateIndex)
    countriescluster = bi.get_cluster(countries, countryIndex)
    streetscluster = bi.get_cluster(streets, streetIndex)
    organizationscluster = bi.get_cluster(organizations, organizationIndex)
    usernamescluster = bi.get_cluster(usernames, usernameIndex)
    
    #post processing for profession
    
    hc.logfile(professionlogclustering, professionscluster)
    hc.logfile(citylogclustering, citiescluster)
    hc.logfile(statelogclustering, statescluster)
    hc.logfile(countrylogclustering, countriescluster)
    hc.logfile(streetlogclustering, streetscluster)
    hc.logfile(organizationlogclustering, organizationscluster)
    hc.logfile(usernamelogclustering, usernamescluster)
    docslogclustering.close()
    hospitallogclustering.close()
    

    print('log cluster success')
    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/newemr.owl").load()
    # print(onto.Doctor)
    
    # for doctor in onto.Doctor.instances():
    #     print(doctor.name)
    #     break

    doctorsName = [doctor.name for doctor in onto.Doctor.instances()]
    professionsName = [profession.name for profession in onto.Profession.instances()]
    citysName = [city.name for city in onto.City.instances()]
    statesName = [state.name for state in onto.State.instances()]
    streetsName = [street.name for street in onto.Street.instances()]
    countrysName = [country.name for country in onto.Country.instances()]
    hospitalsName = [hospital.name for hospital in onto.Hospital.instances()]
    organizationsName = [organization.name for organization in onto.Organization.instances()]
    usernamesName = [username.name for username in onto.Username.instances()]

    hc = BiKmeanClustering()

    doctorsmaxlength = hc.getmaxlengthabb(doctorsName, is_username = False)
    professionsmaxlength = hc.getmaxlengthabb(professionsName, is_username = False)
    citysmaxlength = hc.getmaxlengthabb(citysName, is_username = False)
    statesmaxlength = hc.getmaxlengthabb(statesName, is_username = False)
    streetsmaxlength = hc.getmaxlengthabb(streetsName, is_username = False)
    countrysmaxlength = hc.getmaxlengthabb(countrysName, is_username = False)
    hospitalsmaxlength = hc.getmaxlengthabb(hospitalsName, is_username = False)
    organizationsmaxlength = hc.getmaxlengthabb(organizationsName, is_username = False)
    usernamesmaxlength = hc.getmaxlengthabb(usernamesName, is_username = True)


    constructionbikmeancluster(hc, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv=True)

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


