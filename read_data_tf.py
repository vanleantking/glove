import tf_glove
import re
import nltk
from collections import Counter, defaultdict
import os
import sys
import bs4 as bs
import unicodedata
from string import punctuation
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bikmeans import BiKMeans
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from owlready2 import *


BASE_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/data'


def load_corpus():
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    texts = []  # list of text samples
    doc = []
    words_train = 0
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
                        sentence = "".join(c for c in sent if c not in ('!','.',':', '-', '\'s', '+', '_', '(', ')', '*', '&', '#', ';', '?', '>', '<', '%'))
                        process_words = " ".join(word for word in sentence.split() if word not in stopwords.words('english'))
                        words_arrays = word_tokenize(process_words.strip().lower())
                        words_train += len(words_arrays)
                        texts.append(words_arrays)
    return texts, words_train

def old_learning_phrase(corpus, corpus_size, word_phrases = 4, min_count = 5, threshold = 100):
    file = open("learning_phrase.txt","w") 
    word_counts = Counter()
    loop = 1
    while(loop < word_phrases):
        loop +=1
        print(loop)
        if (loop == 2):
            threshold = 200
        for line in corpus:
            word_counts.update(line)
            for i in range(len(line)):
                if i + 1 < len(line):
                    word_counts.update([line[i]+"_"+line[i+1]])

        for line in corpus:
            i = 0
            while (i < len(line) - 1):
                if (word_counts[line[i]]) < min_count or (word_counts[line[i+1]]) < min_count :
                    i += 1
                else:
                    word_score = (word_counts[line[i]+"_"+line[i+1]] - min_count) / float(word_counts[line[i]]) / float(word_counts[line[i+1]]) * corpus_size;
                    
                    if word_score > threshold:
                        file.write(str(word_score) + " " + line[i] + " " + line[i+1] + " " + str(word_counts[line[i]+"_"+line[i+1]]) + " " + str(min_count) + " " + str(word_counts[line[i]]) + " " + str(word_counts[line[i+1]]) + " " + str(corpus_size))
                        file.write('\n')
                        line.insert(i+1, line[i]+"_"+line[i+1])
                        i = i + 2
                    else:
                        del word_counts[line[i]+"_"+line[i+1]]
                        i = i+1
                    
    return corpus

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
    


    
if __name__ == '__main__':
    
    onto = get_ontology("file:///media/vanle/Studying/python/readOntology/emr.owl").load()
    file = open("phrase_corpus.txt","w") 
    print('Pre-processing corpus')
    corpus, corpus_size = load_corpus()
    corpus = learning_phrase(corpus, corpus_size)
    print('learning phrase completed')
    for text in corpus:
        file.write(' '.join(text))
        file.write('\n')
    file.close()

    model = tf_glove.GloVeModel(embedding_size=100, context_size=2)
    model.fit_to_corpus(corpus)
    model.train(num_epochs=150, log_dir="log/example", summary_batch_interval=1000)


    patitens = []
    for i in onto.Patient.instances():
        patitens.append(model.embeded_phrases(i.name))

    bikmeans = KMeans(n_clusters=2, random_state=0).fit(patitens)
    print(bikmeans.labels_)

    doctors = []
    l_doctors = []
    for i in onto.Doctor.instances():
        doctors.append(i.name)
        l_doctors.append(model.embeded_phrases(i.name))

    docs = KMeans(n_clusters=10, random_state=0).fit(l_doctors)
    print(cosine_similarity([model.embeded_phrases("Yosef Q Ullrich")], [model.embeded_phrases("Yosef Ullrich")]))
    print(cosine_similarity([model.embeded_phrases("Y Ullrich")], [model.embeded_phrases("Y. Ullrich")]))
    print(cosine_similarity([model.embeded_phrases("Y Ullrich")], [model.embeded_phrases("Ullrich")]))
    print(cosine_similarity([model.embeded_phrases("Yosef Q Ullrich")], [model.embeded_phrases("Ullrich")]))
    print(cosine_similarity([model.embeded_phrases("Y Ullrich")], [model.embeded_phrases("Y. Ullrich")]))
    print(cosine_similarity([model.embeded_phrases("Ullysses B. Gilbert")], [model.embeded_phrases("Ullysses Gilbert")]))
    print(cosine_similarity([model.embeded_phrases("Ullysses B. Gilbert")], [model.embeded_phrases("Ullysses Gilbert")]))
    print(cosine_similarity([model.embeded_phrases("Ullysses B. Gilbert")], [model.embeded_phrases("Gilbert")]))
    print(cosine_similarity([model.embeded_phrases("Gilbert")], [model.embeded_phrases("Ullysses Gilbert")]))
    print(cosine_similarity([model.embeded_phrases("Yosef Q Ullrich")], [model.embeded_phrases("Herman N. Weller")]))
    print(cosine_similarity(l_doctors))
    print(model.embeded_phrases("Betty Kaitlin Wood"))
    # print(l_doctors)
    print(doctors)
    print(docs.labels_)
    print("Betty Kaitlin Wood: ", cosine_similarity([model.embeded_phrases("Betty Kaitlin Wood")], l_doctors))
    print("Herman N. Weller: ", cosine_similarity([model.embeded_phrases("Herman N. Weller")], l_doctors))
    print("Yosef Q Ullrich: ", cosine_similarity([model.embeded_phrases("Yosef Q Ullrich")], l_doctors))
    
    model.generate_tsne(path='log/tsne')
    # print("doctorrrrrrrrrrrrrrrrr: ", doctors)


    

