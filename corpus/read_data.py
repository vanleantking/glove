import os
import sys
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import bs4 as bs
from unidecode import unidecode
import argparse
import pprint
from glove import Glove
from glove import Corpus
import gensim


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

BASE_DIR_DATA = '/media/vanle/Studying/python/word2vec/glove/data'
def load_data():

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(BASE_DIR_DATA)):
        path = os.path.join(BASE_DIR_DATA, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='UTF-8')
                t = f.read()
                text = bs.BeautifulSoup(t, 'xml')
                data = text.TEXT.string
                texts.append(unidecode(data))
                f.close()
                labels.append(label_id)
    return texts, labels, labels_index

def get_data(texts):
    # We don't want to do a dictionary construction pass.
    corpus = []
    for text in texts:
        for line in text:
            sequences = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=" ")
            yield sequences
    


    
if __name__ == '__main__':
    
    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Fit a GloVe model.')

    parser.add_argument('--create', '-c', action='store',
                        default=None,
                        help=('The filename of the corpus to pre-process. '
                              'The pre-processed corpus will be saved '
                              'and will be ready for training.'))
    parser.add_argument('-wiki', '-w', action='store_true',
                        default=False,
                        help=('Assume the corpus input file is in the '
                              'Wikipedia dump format'))
    parser.add_argument('--train', '-t', action='store',
                        default=0,
                        help=('Train the GloVe model with this number of epochs.'
                              'If not supplied, '
                              'We\'ll attempt to load a trained model'))
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help=('Number of parallel threads to use for training'))
    parser.add_argument('--query', '-q', action='store',
                        default='',
                        help='Get closes words to this word.')
    args = parser.parse_args()


    if args.create:
        # Build the corpus dictionary and the cooccurrence matrix.
        print('Pre-processing corpus')
        
        texts, labels, labels_index = load_data()
        print('Found %s texts.' % len(texts))
        # print(texts[10])
        
        corpus = get_data(texts)

        corpus_model = Corpus()
        corpus_model.fit(corpus, window=10)
        corpus_model.save('corpus.model')
        
        print('Dict size: %s' % len(corpus_model.dictionary))
        print('Collocations: %s' % corpus_model.matrix.nnz)

    if args.train:
        # Train the GloVe model and save it to disk.

        if not args.create:
            # Try to load a corpus from disk.
            print('Reading corpus statistics')
            corpus_model = Corpus.load('corpus.model')

            print('Dict size: %s' % len(corpus_model.dictionary))
            print('Collocations: %s' % corpus_model.matrix.nnz)

        print('Training the GloVe model')

        glove = Glove(no_components=100, learning_rate=0.05)
        glove.fit(corpus_model.matrix, epochs=int(args.train),
                  no_threads=args.parallelism, verbose=True)
        glove.add_dictionary(corpus_model.dictionary)

        glove.save('glove.model')

    if args.query:
        # Finally, query the model for most similar words.
        if not args.train:
            print('Loading pre-trained GloVe model')
            glove = Glove.load('glove.model')

        print('Querying for %s' % args.query)
        pprint.pprint(glove.most_similar(args.query, number=10))

