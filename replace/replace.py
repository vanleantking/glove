import os
import sys
import bs4 as bs
import unicodedata
from owlready2 import *

BASE_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/data'


onto = get_ontology("file:///media/vanle/Studying/python/word2vec/glove/thesis/clustering/newemr.owl").load()
def read_file():

    for name in sorted(os.listdir(BASE_DIR_DATA)):
        path = os.path.join(BASE_DIR_DATA, name)
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                
                f = open(fpath, encoding='UTF-8')
                t = f.read()
                text = bs.BeautifulSoup(t, 'xml')
                data = text.TEXT.string
                tags = text.TAGS

                tmp = unicodedata.normalize('NFKD', data).encode('ascii','ignore').decode("utf-8")
                dept = 0
                medicalData = load_onto_data(fname)

                # #replace fake infor to original file
                # for data in sorted(medicalData.keys()):
                #     tmp, dept = replacement(tmp, dept = dept, start = int(medicalData[data][0]), end = int(medicalData[data][1]), replacement=medicalData[data][2])
                
                for data in sorted(medicalData.keys()):
                    # print(data)
                    for tag in tags:
                        if tag.name and data == int(tag['start']):                            
                            tag['fakeinfo'] = medicalData[data][2]
                            # print(tag.name, tag['start'])
                            print(tag)
                
                break
            break


def get_medial_record(fname):

    for medicalRecord in onto.MedicalRecord.instances():
        if medicalRecord.name == fname:
            return medicalRecord

def load_onto_data(fname):
    medicalRecord = get_medial_record(fname)
    onto_datas = {}


    onto_datas = {int(age.hasStartPosition[0]) : [age.hasStartPosition[0], age.hasEndPosition[0], age.hasCloneInfo[0]] \
    for age in medicalRecord.has_record_at_age}

    onto_datas.update({int(contact.hasStartPosition[0]) : [contact.hasStartPosition[0], contact.hasEndPosition[0], contact.hasCloneInfo[0]] \
    for contact in medicalRecord.has_contact})

    onto_datas.update({int(date.hasStartPosition[0]) : [date.hasStartPosition[0], date.hasEndPosition[0], date.hasCloneInfo[0]] \
    for date in medicalRecord.has_observation_date})

    onto_datas.update({int(device.hasStartPosition[0]) : [device.hasStartPosition[0], device.hasEndPosition[0], device.hasCloneInfo[0]] \
    for device in medicalRecord.was_use_device})

    onto_datas.update({int(doctor.hasStartPosition[0]) : [doctor.hasStartPosition[0], doctor.hasEndPosition[0], doctor.hasCloneInfo[0]] \
    for doctor in medicalRecord.doctor_dianose})

    onto_datas.update({int(hospital.hasStartPosition[0]) : [hospital.hasStartPosition[0], hospital.hasEndPosition[0], hospital.hasCloneInfo[0]] \
    for hospital in medicalRecord.record_from_hospital})

    onto_datas.update({int(bioID.hasStartPosition[0]) : [bioID.hasStartPosition[0], bioID.hasEndPosition[0], bioID.hasCloneInfo[0]] \
    for bioID in medicalRecord.has_unique_BioID})

    onto_datas.update({int(healthplan.hasStartPosition[0]) : [healthplan.hasStartPosition[0], healthplan.hasEndPosition[0], healthplan.hasCloneInfo[0]] \
    for healthplan in medicalRecord.has_healthPlan})

    onto_datas.update({int(idNum.hasStartPosition[0]) : [idNum.hasStartPosition[0], idNum.hasEndPosition[0], idNum.hasCloneInfo[0]] \
    for idNum in medicalRecord.has_unique_IDNum})

    onto_datas.update({int(city.hasStartPosition[0]) : [city.hasStartPosition[0], city.hasEndPosition[0], city.hasCloneInfo[0]] \
    for city in medicalRecord.has_city})

    onto_datas.update({int(country.hasStartPosition[0]) : [country.hasStartPosition[0], country.hasEndPosition[0], country.hasCloneInfo[0]] \
    for country in medicalRecord.has_country})

    onto_datas.update({int(locationOther.hasStartPosition[0]) : [locationOther.hasStartPosition[0], locationOther.hasEndPosition[0], locationOther.hasCloneInfo[0]] \
    for locationOther in medicalRecord.has_locationother})

    onto_datas.update({int(organization.hasStartPosition[0]) : [organization.hasStartPosition[0], organization.hasEndPosition[0], organization.hasCloneInfo[0]] \
    for organization in medicalRecord.has_organization})

    onto_datas.update({int(state.hasStartPosition[0]) : [state.hasStartPosition[0], state.hasEndPosition[0], state.hasCloneInfo[0]] \
    for state in medicalRecord.has_state})
    
    onto_datas.update({int(street.hasStartPosition[0]) : [street.hasStartPosition[0], street.hasEndPosition[0], street.hasCloneInfo[0]] \
    for street in medicalRecord.has_street})
    
    onto_datas.update({int(zipNumber.hasStartPosition[0]) : [zipNumber.hasStartPosition[0], zipNumber.hasEndPosition[0], zipNumber.hasCloneInfo[0]] \
    for zipNumber in medicalRecord.has_zip})
    
    onto_datas.update({int(patient.hasStartPosition[0]) : [patient.hasStartPosition[0], patient.hasEndPosition[0], patient.hasCloneInfo[0]] \
    for patient in medicalRecord.have_collect_patients})
    
    onto_datas.update({int(username.hasStartPosition[0]) : [username.hasStartPosition[0], username.hasEndPosition[0], username.hasCloneInfo[0]] \
    for username in medicalRecord.has_username})
    
    onto_datas.update({int(profession.hasStartPosition[0]) : [profession.hasStartPosition[0], profession.hasEndPosition[0], profession.hasCloneInfo[0]] \
    for profession in medicalRecord.job_position})


    if len(medicalRecord.hasCloneInfo) > 0:
        onto_datas.update({int(medicalRecord.hasStartPosition[0]) : [medicalRecord.hasStartPosition[0], medicalRecord.hasEndPosition[0], medicalRecord.hasCloneInfo[0]]})

    # sortkeys = sorted(onto_datas.keys())
    # for key in sortkeys:

    # print(sortkeys)
    return onto_datas

def replacement(text, end, start = 0, dept = 0, replacement=''):
    print(type(dept), type(end), type(start), type(dept), type(replacement))
    s = '%s%s%s'%(text[:start + dept],replacement,text[end+dept:])
    dept = dept + len(replacement) - (end - start)
    return s, dept


if __name__ == '__main__':
    read_file()
    # print(load_onto_data('362-04.xml'))

            
