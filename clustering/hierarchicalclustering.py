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
from nameprocessing import PreProcessingText, PostProcessing, PostHospitalClusterProcessing, \
PostNameClusterProcessing, PostLocationClusteringProcessing

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from corpus.tf_glove import GloVeModel
from corpus.read_data_tf import load_corpus, learning_phrase
from clustering import Clustering
from replace.generateinfo import FakerInfo
from owlready2 import *



RESULT_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/thesis/clustering/result/hierarchicalclustering/'

#location clustering log path for post clustering and hierarchy
doctorclusterpath = os.path.join(RESULT_DIR_DATA, 'doctorcluster.txt')
patientclusterpath = os.path.join(RESULT_DIR_DATA, 'patientcluster.txt')
professionclusterpath = os.path.join(RESULT_DIR_DATA, 'professioncluster.txt')
cityclusterpath = os.path.join(RESULT_DIR_DATA, 'citycluster.txt')
stateclusterpath = os.path.join(RESULT_DIR_DATA, 'statecluster.txt')
streetclusterpath = os.path.join(RESULT_DIR_DATA, 'streetcluster.txt')
organizationclusterpath = os.path.join(RESULT_DIR_DATA, 'organizationcluster.txt')
hospitalclusterpath = os.path.join(RESULT_DIR_DATA, 'hospitalcluster.txt')
countryclusterpath = os.path.join(RESULT_DIR_DATA, 'countrycluster.txt')
usernameclusterpath = os.path.join(RESULT_DIR_DATA, 'usernamecluster.txt')


#location clustering log path for  hierarchy
aa_not_postdoctorclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postdoctorcluster.txt')
aa_not_postpatientclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postpatientcluster.txt')
aa_not_postprofessionclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postprofessioncluster.txt')
aa_not_postcityclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postcitycluster.txt')
aa_not_poststateclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_poststatecluster.txt')
aa_not_poststreetclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_poststreetcluster.txt')
aa_not_postorganizationclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postorganizationcluster.txt')
aa_not_posthospitalclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_posthospitalcluster.txt')
aa_not_postcountryclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postcountrycluster.txt')
aa_not_postusernameclusterpath = os.path.join(RESULT_DIR_DATA, 'aa_not_postusernamecluster.txt')

class HierachicalClustering(Clustering):

    def __init__(self):
        super().__init__()


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

        distance = 0

        for Z in Zdocs:
            distance += Z[2]

        weighted_distance = distance / len(Zdocs)

        clustersdocs = fcluster(Zdocs, weighted_distance, criterion='distance')

        clusters_number =  len(np.unique(clustersdocs))

        docscluster = {i : [] for i in range(1, clusters_number+1)}

        for index, docIndex in enumerate(dataIndex):
            if np.array_equal(docIndex['value'],datas[index]):
                docscluster[clustersdocs[index]].append(docIndex['name'] )

        return docscluster, clusters_number

    def postclustering(self, clusters, clusters_number, processing_type):
        clusters, clusters_number = processing_type.splitcluster(clusters)
        for i in range(clusters_number):
            for j in range (i+1, clusters_number):
                if i in clusters.keys() and j in clusters.keys():
                    merged = processing_type.mergecluster(clusters[i], clusters[j])
                    if merged:
                        try:
                            clusters[i].extend(clusters[j])
                            del clusters[j]
                        except:
                            pass
        return clusters



def constructioncluster(hc, onto, patientsmaxlength, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv = False, is_post = True):
    if is_post == True:
        patientslogclustering = open(patientclusterpath,"w")
        docslogclustering = open(doctorclusterpath,"w")
        professionlogclustering = open(professionclusterpath,"w")
        citylogclustering = open(cityclusterpath,"w")
        statelogclustering = open(stateclusterpath,"w")
        streetlogclustering = open(streetclusterpath,"w")
        countrylogclustering = open(countryclusterpath,"w")
        hospitallogclustering = open(hospitalclusterpath,"w")
        organizationlogclustering = open(organizationclusterpath,"w")
        usernamelogclustering = open(usernameclusterpath,"w")
    else:
        patientslogclustering = open(aa_not_postpatientclusterpath,"w")
        docslogclustering = open(aa_not_postdoctorclusterpath,"w")
        professionlogclustering = open(aa_not_postprofessionclusterpath,"w")
        citylogclustering = open(aa_not_postcityclusterpath,"w")
        statelogclustering = open(aa_not_poststateclusterpath,"w")
        streetlogclustering = open(aa_not_poststreetclusterpath,"w")
        countrylogclustering = open(aa_not_postcountryclusterpath,"w")
        hospitallogclustering = open(aa_not_posthospitalclusterpath,"w")
        organizationlogclustering = open(aa_not_postorganizationclusterpath,"w")
        usernamelogclustering = open(aa_not_postusernameclusterpath,"w")

    is_abbrv = False
    for patientRecords in onto.PatientRecord.instances():

        #data per patient
        patients = []
        patientIndex = []
        doctors = []
        doctorIndex = []       
        hospitalIndex = []
        hospitals = []
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
        
        if len(patientRecords.have_collect_patients) > 1:
            patientRecord = [hc.getword2vectabb(patient.hasName[0], patientsmaxlength, is_abbrv) for patient in patientRecords.have_collect_patients if np.isnan(hc.get_model().embeded_phrases(patient.hasName[0])).any() == False]
            patientIndex.extend([{"name": patient, "value": hc.getword2vectabb(patient.hasName[0], patientsmaxlength, is_abbrv)} for patient in patientRecords.have_collect_patients if np.isnan(hc.get_model().embeded_phrases(patient.hasName[0])).any() == False])
            patients.extend(patientRecord)

        #get data for each medical record
        for medicalRecord in patientRecords.was_recorded_at:
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
                stateRecord = [hc.getword2vectabb(state.hasLocation[0], statesmaxlength, is_abbrv) for state in medicalRecord.has_state if np.isnan(hc.get_model().embeded_phrases(state.hasLocation[0])).any() == False]
                stateIndex.extend([{"name": state, "value": hc.getword2vectabb(state.hasLocation[0], statesmaxlength, is_abbrv)} for state in medicalRecord.has_state if np.isnan(hc.get_model().embeded_phrases(state.hasLocation[0])).any() == False])
                states.extend(stateRecord)

            if len(medicalRecord.has_country) > 0:
                countryRecord = [hc.getword2vectabb(country.hasLocation[0], countrysmaxlength, is_abbrv) for country in medicalRecord.has_country if np.isnan(hc.get_model().embeded_phrases(country.hasLocation[0])).any() == False]
                countryIndex.extend([{"name": country, "value": hc.getword2vectabb(country.hasLocation[0], countrysmaxlength, is_abbrv)} for country in medicalRecord.has_country if np.isnan(hc.get_model().embeded_phrases(country.hasLocation[0])).any() == False])
                countries.extend(countryRecord)

            if len(medicalRecord.has_organization) > 0:
                organizationRecord = [hc.getword2vectabb(organization.hasLocation[0], organizationsmaxlength, is_abbrv) for organization in medicalRecord.has_organization if np.isnan(hc.get_model().embeded_phrases(organization.hasLocation[0])).any() == False]
                organizationIndex.extend([{"name": organization, "value": hc.getword2vectabb(organization.hasLocation[0], organizationsmaxlength, is_abbrv)} for organization in medicalRecord.has_organization if np.isnan(hc.get_model().embeded_phrases(organization.hasLocation[0])).any() == False])
                organizations.extend(organizationRecord)

            if len(medicalRecord.has_street) > 0:
                streetRecord = [hc.getword2vectabb(street.hasLocation[0], streetsmaxlength, is_abbrv) for street in medicalRecord.has_street if np.isnan(hc.get_model().embeded_phrases(street.hasLocation[0])).any() == False]
                streetIndex.extend([{"name": street, "value": hc.getword2vectabb(street.hasLocation[0], streetsmaxlength, is_abbrv)} for street in medicalRecord.has_street if np.isnan(hc.get_model().embeded_phrases(street.hasLocation[0])).any() == False])
                streets.extend(streetRecord)

            if len(medicalRecord.record_from_hospital) > 0:
                hospitalRecord = [hc.getword2vectabb(hospital.hasName[0], hospitalsmaxlength, is_abbrv) for hospital in medicalRecord.record_from_hospital if np.isnan(hc.get_model().embeded_phrases(hospital.hasName[0])).any() == False]
                hospitalIndex.extend([{"name": hospital, "value": hc.getword2vectabb(hospital.hasName[0], hospitalsmaxlength, is_abbrv)} for hospital in medicalRecord.record_from_hospital if np.isnan(hc.get_model().embeded_phrases(hospital.hasName[0])).any() == False])
                hospitals.extend(hospitalRecord)

            if len(medicalRecord.has_username) > 0:
                usernameRecord = [hc.getword2vectabb(username.hasName[0], usernamesmaxlength, is_abbrv) for username in medicalRecord.has_username if np.isnan(hc.get_model().embeded_phrases(username.hasName[0])).any() == False]
                usernameIndex.extend([{"name": username, "value": hc.getword2vectabb(username.hasName[0], usernamesmaxlength, is_abbrv)} for username in medicalRecord.has_username if np.isnan(hc.get_model().embeded_phrases(username.hasName[0])).any() == False])
                usernames.extend(usernameRecord)


        ###DOCTORS###
        # docscluster, clusters_number = hc.hierarchical(doctors, doctorIndex, 0.45)
        
        #post processing for doctors
        docslogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))

        if len(doctors) > 1:
            docscluster, clusters_number = hc.hierarchical(doctors, doctorIndex, 0.35)
            pncp = PostNameClusterProcessing()
            if is_post == True:
                docscluster = hc.postclustering(docscluster, clusters_number, pncp)
        elif len(doctors) == 1:
            docscluster = {1: []}
            docscluster[1].append(doctorIndex[0]['name'])
        
        #generate fake infor for doctor phi instances
        if len(doctors) > 0:
            for cluster, docsinstances in docscluster.items():
                fake_doctor = faker.generate_name()
                for doc in docsinstances:
                    doc.hasCloneInfo.append(fake_doctor)

            hc.logdocsfile(docslogclustering, docscluster, doctorIndex)




        ###PATIENTS###

        # post procressing for patients
        patientslogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(patients) > 1:
            patientscluster, clusters_patient_number = hc.hierarchical(patients, patientIndex, 0.35)
            pnpcp = PostNameClusterProcessing()
            if is_post == True:
                patientscluster = hc.postclustering(patientscluster, clusters_patient_number, pnpcp)
        elif len(patients) == 1:
            patientscluster = {1: []}
            patientscluster[1].append(patientIndex[0]['name'])


        #generate fake infor for patient phi instances
        if len(patients) > 0:
            for cluster, patientsinstances in patientscluster.items():
                fake_patient = faker.generate_name()
                for patientinstance in patientsinstances:
                    patientinstance.hasCloneInfo.append(fake_patient)
            hc.logdocsfile(patientslogclustering, patientscluster, patientIndex)



        ###HOSPITALS###
        #post procressing for hospital
        hospitallogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(hospitals) > 1:
            hospitalscluster, clusters_hospital_number = hc.hierarchical(hospitals, hospitalIndex, 0.5)
            phcp = PostHospitalClusterProcessing()
            if is_post == True:
                hospitalscluster = hc.postclustering(hospitalscluster, clusters_hospital_number, phcp)
        elif len(hospitals) == 1:
            hospitalscluster = {1 : []}
            hospitalscluster[1].append(hospitalIndex[0]['name'])

        #generate fake infor for hospital phi instances
        if len(hospitals) > 0:
            for cluster, hospitalsinstances in hospitalscluster.items():
                fake_hospital = faker.generate_hospital()
                for hospitalinstance in hospitalsinstances:
                    hospitalinstance.hasCloneInfo.append(fake_hospital)
            hc.logdocsfile(hospitallogclustering, hospitalscluster, hospitalIndex)



        ###PROFESSIONS###
        #post procressing for profession
        professionlogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(professions) > 1:
            professionscluster, clusters_profession_number = hc.hierarchical(professions, professionIndex, 0.5)
        elif len(professions) == 1:
            professionscluster = {1 : []}
            professionscluster[1].append(professionIndex[0]['name'])


        if len(professions) > 0:
            #generate fake infor for profession phi instances
            for cluster, professionsinstances in professionscluster.items():
                fake_profession = faker.generate_profession()
                for professioninstance in professionsinstances:
                    professioninstance.hasCloneInfo.append(fake_profession)
            hc.logdocsfile(professionlogclustering, professionscluster, professionIndex)


        pccp = PostLocationClusteringProcessing()
        
        ###CITIES###
        #post procressing for city
        citylogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(cities) > 1:
            citiescluster, clusters_city_number = hc.hierarchical(cities, cityIndex, 0.5)
            if is_post == True:
                citiescluster = hc.postclustering(citiescluster, clusters_city_number, pccp)
        elif len(cities) == 1:
            citiescluster = {1 : []}
            citiescluster[1].append(cityIndex[0]['name'])

        #generate fake infor for city phi instances
        if len(cities) > 0:
            for cluster, citiesinstances in citiescluster.items():
                fake_city = faker.generate_city()
                for cityinstance in citiesinstances:
                    cityinstance.hasCloneInfo.append(fake_city)
            hc.logdocsfile(citylogclustering, citiescluster, cityIndex)




        ###STATES###
        #post procressing for state
        statelogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(states) > 1:
            statescluster, clusters_state_number = hc.hierarchical(states, stateIndex, 0.5)
            if is_post == True:
                statescluster = hc.postclustering(statescluster, clusters_state_number, pccp)
        elif len(states) == 1:
            statescluster = {1 : []}
            statescluster[1].append(stateIndex[0]['name'])

        #generate fake infor for state phi instances
        if len(states) > 0:
            for cluster, statesinstances in statescluster.items():
                fake_state = faker.generate_state()
                for stateinstance in statesinstances:
                    stateinstance.hasCloneInfo.append(fake_state)
            hc.logdocsfile(statelogclustering, statescluster, stateIndex)




        ###COUNTRIES###
        #post procressing for country instance
        countrylogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(countries) > 1:
            countriescluster, clusters_country_number = hc.hierarchical(countries, countryIndex, 0.75)
            if is_post == True:
                countriescluster = hc.postclustering(countriescluster, clusters_country_number, pccp)
        elif len(countries) == 1:
            countriescluster = {1 : []}
            countriescluster[1].append(countryIndex[0]['name'])

        #generate fake infor for country phi instances
        if len(countries) > 0:
            for cluster, countriesinstances in countriescluster.items():
                fake_country = faker.generate_country()
                for countryinstance in countriesinstances:
                    countryinstance.hasCloneInfo.append(fake_country)
            hc.logdocsfile(countrylogclustering, countriescluster, countryIndex)




        ###STREETS###
        #post procressing for street instance
        streetlogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(streets) > 1:
            streetscluster, clusters_street_number = hc.hierarchical(streets, streetIndex, 0.5)
            if is_post == True:
                streetscluster = hc.postclustering(streetscluster, clusters_street_number, pccp)
        elif len(streets) == 1:
            streetscluster = {1 : []}
            streetscluster[1].append(streetIndex[0]['name'])

        #generate fake infor for street phi instances
        fake_streets = []
        if len(streets) > 0:
            for cluster, streetsinstances in streetscluster.items():
                fake_streets = faker.generate_street(streetsinstances)
                for i in range(len(streetsinstances)):
                    streetsinstances[i].hasCloneInfo.append(fake_streets[i])
            hc.logdocsfile(streetlogclustering, streetscluster, streetIndex)





        ###ORGANIZATIONS###
        #post procressing for organization instance
        organizationlogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(organizations) > 1:
            organizationscluster, clusters_organization_number = hc.hierarchical(organizations, organizationIndex, 0.5)
            if is_post == True:
                organizationscluster = hc.postclustering(organizationscluster, clusters_organization_number, pccp)
        elif len(organizations) == 1:
            organizationscluster = {1 : []}
            organizationscluster[1].append(organizationIndex[0]['name'])

        #generate fake infor for organization phi instances
        if len(organizations) > 0:
            for cluster, organizationsinstances in organizationscluster.items():
                fake_organization = faker.generate_company()
                for organizationinstance in organizationsinstances:
                    organizationinstance.hasCloneInfo.append(fake_organization)
            hc.logdocsfile(organizationlogclustering, organizationscluster, organizationIndex)





        ###USERNAMES###
        #post procressing for username instance
        usernamelogclustering.write("result patienttttttttttttttttttttttttttttttttttttttttttttttttt: " + str(patientRecords.hasRecordName[0]))
        if len(usernames) > 1:
            usernamescluster, clusters_username_number = hc.hierarchical(usernames, usernameIndex, 0.5)
        elif len(usernames) == 1:
            usernamescluster = {1 : []}
            usernamescluster[1].append(usernameIndex[0]['name'])

        #generate fake infor for username phi instances
        if len(usernames) > 0:
            for cluster, usernamesinstances in usernamescluster.items():
                fake_username = faker.generate_username()
                for usernameinstance in usernamesinstances:
                    usernameinstance.hasCloneInfo.append(fake_username)
            hc.logdocsfile(usernamelogclustering, usernamescluster, usernameIndex) 
    
    
    
    
    
    
    docslogclustering.close()
    hospitallogclustering.close()
    patientslogclustering.close()
    professionlogclustering.close()
    citylogclustering.close()
    statelogclustering.close()
    streetlogclustering.close()
    countrylogclustering.close()
    organizationlogclustering.close()
    usernamelogclustering.close()
    

    print('log cluster success')
    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/newemr.owl").load()
    faker = FakerInfo()

    datesName = [date for date in onto.Date.instances()]
    agesName = [age for age in onto.Age.instances()]
    phonesName = [phone for phone in onto.Phone.instances()]
    faxsName = [fax for fax in onto.Fax.instances()]
    emailsName = [email for email in onto.Email.instances()]
    urlsName = [url for url in onto.URL.instances()]
    medicalRecordsName = [medicalRecord for medicalRecord in onto.MedicalRecord.instances() if len(medicalRecord.hasMedicalRecordID) > 0]
    IDNumsName = [idNum for idNum in onto.IDNum.instances()]
    devicesName = [device for device in onto.Device.instances()]
    bioIDsName = [bioID for bioID in onto.BioID.instances()]
    healthPlansName = [healthPlan for healthPlan in onto.HealthPlan.instances()]
    zipsName = [zips for zips in onto.Zip.instances()]
    patientsName = [patient.hasName[0] for patientRecords in onto.PatientRecord.instances() for patient in patientRecords.have_collect_patients]
    doctorsName = [doctor.hasName[0] for doctor in onto.Doctor.instances()]
    usernamesName = [username.hasName[0] for username in onto.Username.instances()]
    hospitalsName = [hospital.hasName[0] for hospital in onto.Hospital.instances()]
    citysName = [city.hasLocation[0] for city in onto.City.instances()]
    statesName = [state.hasLocation[0] for state in onto.State.instances()]
    streetsName = [street.hasLocation[0] for street in onto.Street.instances()]
    organizationsName = [organization.hasLocation[0] for organization in onto.Organization.instances()]
    countrysName = [country.hasLocation[0] for country in onto.Country.instances()]
    locationOthersName = [locationOther.has_state[0] for locationOther in onto.LocationOther.instances()]
    professionsName = [profession.jobName[0] for profession in onto.Profession.instances()]

    

    # save fake infor for date
    for date in datesName:
        fk_date = faker.generate_date(date.hasDate[0])
        date.hasCloneInfo.append(fk_date)


    # save fake infor for date
    dictAges = dict.fromkeys(agesName)
    for k in dictAges.keys():
        dictAges[k] = []    

    for age in agesName:
        dictAges[age].append(age)

    for k, ages in dictAges.items():
        fk_age = faker.generate_age(k.hasAge[0])
        for age in ages:
            age.hasCloneInfo.append(fk_age)



    # save fake infor for phone
    dictPhones = dict.fromkeys(phonesName)
    for k in dictPhones.keys():
        dictPhones[k] = []    

    for phone in phonesName:
        dictPhones[phone].append(phone)

    for k, phones in dictPhones.items():
        fk_phone = faker.generate_phone()
        for phone in phones:
            phone.hasCloneInfo.append(fk_phone)


    #save fake infor for phone
    dictFaxs = dict.fromkeys(faxsName)
    for k in dictFaxs.keys():
        dictFaxs[k] = []    

    for fax in faxsName:
        dictFaxs[fax].append(fax)

    for k, faxes in dictFaxs.items():
        fk_fax = faker.generate_fax()
        for fax in faxes:
            fax.hasCloneInfo.append(fk_fax)



    #save fake infor for phone
    dictEmails = dict.fromkeys(emailsName)
    for k in dictEmails.keys():
        dictEmails[k] = []    

    for email in emailsName:
        dictEmails[email].append(email)

    for k, faxes in dictEmails.items():
        fk_email = faker.generate_email()
        for email in faxes:
            email.hasCloneInfo.append(fk_email)



    #save fake infor for phone
    dictURLs = dict.fromkeys(urlsName)
    for k in dictURLs.keys():
        dictURLs[k] = []

    for url in urlsName:
        dictURLs[url].append(url)

    for k, urls in dictURLs.items():
        fk_url = faker.generate_URL()
        for url in urls:
            url.hasCloneInfo.append(fk_url)

    



    #save fake infor for phone
    dictIDNums = dict.fromkeys(IDNumsName)
    for k in dictIDNums.keys():
        dictIDNums[k] = []

    for idNum in IDNumsName:
        dictIDNums[idNum].append(idNum)

    for k, IDNums in dictIDNums.items():
        fk_idNum = faker.generate_IDNum()
        for idNum in IDNums:
            idNum.hasCloneInfo.append(fk_idNum)



    #save fake infor for phone
    dictdevicesName = dict.fromkeys(devicesName)
    for k in dictdevicesName.keys():
        dictdevicesName[k] = []

    for device in devicesName:
        dictdevicesName[device].append(device)

    for k, devices in dictdevicesName.items():
        fk_device = faker.generate_device()
        for device in devices:
            device.hasCloneInfo.append(fk_device)

    

    #save fake infor for phone
    dictBioIDsName = dict.fromkeys(bioIDsName)
    for k in dictBioIDsName.keys():
        dictBioIDsName[k] = []

    for bio in bioIDsName:
        dictBioIDsName[bio].append(bio)

    for k, bioIDs in dictBioIDsName.items():
        fk_bio = faker.generate_BioID()
        for bio in bioIDs:
            bio.hasCloneInfo.append(fk_bio)

    


    #save fake infor for phone
    dictHealthPlansName = dict.fromkeys(healthPlansName)
    for k in dictHealthPlansName.keys():
        dictHealthPlansName[k] = []

    for healthPlan in healthPlansName:
        dictHealthPlansName[healthPlan].append(healthPlan)

    for k, healthPlans in dictHealthPlansName.items():
        fk_healthPlan = faker.generate_healthplan()
        for healthPlan in healthPlans:
            healthPlan.hasCloneInfo.append(fk_healthPlan)

    

    #save fake infor for phone
    dictMedicalRecordsName = dict.fromkeys(medicalRecordsName)
    for k in dictMedicalRecordsName.keys():
        dictMedicalRecordsName[k] = []

    for medicalRecord in medicalRecordsName:
        if len(medicalRecord.hasMedicalRecordID[0]) > 0:
            dictMedicalRecordsName[medicalRecord].append(medicalRecord)

    for k, medicalRecords in dictMedicalRecordsName.items():
        fk_medicalRecord = faker.generate_medicalrecord()
        for medicalRecord in medicalRecords:
            medicalRecord.hasCloneInfo.append(fk_medicalRecord)


    #save fake infor for phone
    dictZipsNames = dict.fromkeys(zipsName)
    for k in dictZipsNames.keys():
        dictZipsNames[k] = []

    for zipNumber in zipsName:
        dictZipsNames[zipNumber].append(zipNumber)

    for k, zipNumbers in dictZipsNames.items():
        fk_zip = faker.generate_zip()
        for zipNumber in zipNumbers:
            zipNumber.hasCloneInfo.append(fk_zip)

    



    hc = HierachicalClustering()

    patientsmaxlength = hc.getmaxlengthabb(patientsName, is_username = False)
    doctorsmaxlength = hc.getmaxlengthabb(doctorsName, is_username = False)
    professionsmaxlength = hc.getmaxlengthabb(professionsName, is_username = False)
    citysmaxlength = hc.getmaxlengthabb(citysName, is_username = False)
    statesmaxlength = hc.getmaxlengthabb(statesName, is_username = False)
    streetsmaxlength = hc.getmaxlengthabb(streetsName, is_username = False)
    countrysmaxlength = hc.getmaxlengthabb(countrysName, is_username = False)
    hospitalsmaxlength = hc.getmaxlengthabb(hospitalsName, is_username = False)
    organizationsmaxlength = hc.getmaxlengthabb(organizationsName, is_username = False)
    usernamesmaxlength = hc.getmaxlengthabb(usernamesName, is_username = True)


    constructioncluster(hc, onto, patientsmaxlength, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv=True)
    onto.save(file="newemr.owl", format = "rdfxml")

    constructioncluster(hc, onto, patientsmaxlength, doctorsmaxlength, professionsmaxlength, citysmaxlength, statesmaxlength, streetsmaxlength, countrysmaxlength, hospitalsmaxlength, organizationsmaxlength, usernamesmaxlength, is_abbrv=True, is_post = False)
    # model.generate_tsne(path='log/tsne')


    

