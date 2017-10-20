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
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import numpy


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
                sents = sent_tokenize(unicodedata.normalize('NFKD', data).encode('ascii','ignore'))
                for sent in sents:
                    if sent.strip():
                        words_arrays = word_tokenize(sent.strip().lower())
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
    
    file = open("phrase_corpus.txt","w") 
    # Build the corpus dictionary and the cooccurrence matrix.
    print('Pre-processing corpus')
    '''
    corpus = load_data()
    
    model = tf_glove.GloVeModel(embedding_size=300, context_size=10)
    model.fit_to_corpus(corpus)
    model.train(num_epochs=1, log_dir="log/example", summary_batch_interval=1000)
    print(model.embedding_for("Oakley"))
    print(model.embeded_phrases("Oakley Narrative"))
    # print(model.embeddings)
    model.generate_tsne(path='log/tsne')

    '''
    corpus, corpus_size = load_corpus()
    corpus = learning_phrase(corpus, corpus_size)
    print('learning phrase completed')
    for text in corpus:
        file.write(' '.join(text))
        file.write('\n')
    file.close()

    model = tf_glove.GloVeModel(embedding_size=300, context_size=4)
    model.fit_to_corpus(corpus)
    model.train(num_epochs=50, log_dir="log/example", summary_batch_interval=1000)
    print(model.embedding_for("oakley"))
    print(model.embeded_phrases("dr oakley"))
    print(model.embeded_phrases("Redwood Area Hospital"))
    

    a = model.embedding_for("oakley")
    b = model.embedding_for("dr")
    d = model.embeded_phrases("dr oakley")
    e = model.embedding_for("clarkfield")
    f = model.embeded_phrases("Oakley narrative")

    c = a + b
    print(cosine_similarity([model.embedding_for("oakley")], [model.embeded_phrases("dr oakley")]))
    print("euclid: ", numpy.linalg.norm(a-d))
    print(cosine_similarity([c], [model.embeded_phrases("dr oakley")]))
    print("euclid: ", numpy.linalg.norm(c-b))
    print(cosine_similarity([model.embedding_for("oakley")], [model.embeded_phrases("Oakley narrative")]))
    print("euclid: ", numpy.linalg.norm(a-f))
    print(cosine_similarity([model.embeded_phrases("dr oakley")], [model.embeded_phrases("Oakley narrative")]))
    print("euclid: ", numpy.linalg.norm(d-f))
    print(cosine_similarity([model.embedding_for("clarkfield")], [model.embeded_phrases("dr oakley")]))
    print("euclid: ", numpy.linalg.norm(e-d))
    print(cosine_similarity([model.embedding_for("clarkfield")], [model.embedding_for("oakley")]))
    print("euclid: ", numpy.linalg.norm(e-a))
    
    model.generate_tsne(path='log/tsne')

    # dist = numpy.linalg.norm(a-b)


    

