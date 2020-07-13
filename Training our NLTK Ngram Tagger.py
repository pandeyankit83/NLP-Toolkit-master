#Detecting Word Shape

import re
def shape(word):
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        return 'number'
    elif re.match('\W+$', word):
        return 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        return 'capitalized'
    elif re.match('[A-Z]+$', word):
        return 'uppercase'
    elif re.match('[a-z]+$', word):
        return 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        return 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        return 'mixedcase'
    elif re.match('__.+__$', word):
        return 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        return 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        return 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        return 'contains-hyphen'
    return 'other'




# Write a function to read the files from the Universal Dependencies Dataset 
#  https://github.com/UniversalDependencies/UD_English

import os
from nltk import conlltags2tree
def read_ud_pos_data(filename):
    """
    Iterate through the Universal Dependencies Corpus Part-Of-Speech data
    Yield sentences one by one, don't load all the data in memory
    """
    current_sentence = []
    with open(filename,  'r', encoding="utf8",) as f:
        for line in f:
            line = line.strip()
            
            # ignore comments
            if line.startswith('#'):
                continue
            # empty line indicates end of sentence
            if not line:
                yield current_sentence
                
                current_sentence=[]
                continue
            annotations = line.split('\t')
            
            # Get only the word and the part of speech
            
            current_sentence.append((annotations[1], annotations[4]))
"""
Use a classifier type that can be trained online: 
    This technique is called Out-Of-Core Learning and implies presenting chunks of data to the classifier, several
times, rather than presenting the whole dataset at once. There are only a few Scikit-Learn classifiers
 that have this functionality, mainly the ones implementing the method. I have used partial_fit method.
 We will use sklearn.linear_model.Perceptron.

• Use a vectorizer able to accommodate new features as they are “discovered”:
This method is called the Hashing Trick. Since we won’t keep the entire 
dataset in memory, it will be impossible to know all the features from the
start. Instead of computing the feature space beforehand, we create a fixed
space that accommodates all our features and assigns a feature to a slot of
the feature space using a hashing function. For this purpose, we will use
the from Scikit-Learn. The Scikit-Learn sklearn.feature_extraction.
"""

# Extending ClassifierBasedTagger
from nltk import ClassifierBasedTagger
from nltk.metrics import accuracy
class ClassifierBasedTaggerBatchTrained(ClassifierBasedTagger):
    def _todataset(self, tagged_sentences):
        classifier_corpus = []
        for sentence in tagged_sentences:
            history = []
            untagged_sentence, tags = zip(*sentence)
            for index in range(len(sentence)):
                featureset = self.feature_detector(untagged_sentence,index, history)
                classifier_corpus.append((featureset, tags[index]))
                history.append(tags[index])
            return classifier_corpus
    
    
    def _train(self, tagged_corpus, classifier_builder, verbose):
        """
        Build a new classifier, based on the given training data
        *tagged_corpus*.
        """
        if verbose:
            print('Constructing training corpus for classifier.')
        self._classifier = classifier_builder(tagged_corpus, lambda sents: self._todataset(sents))


    def evaluate(self, gold):
        dataset = self._todataset(gold)
        featuresets, tags = zip(*dataset)
        predicted_tags = self.classifier().classify_many(featuresets)
        return accuracy(tags, predicted_tags)
    
    
#Feature Extraction Function
from nltk.stem.snowball import SnowballStemmer
#from features import shape

stemmer = SnowballStemmer('english')

def pos_features(sentence, index, history):
    """
    sentence = list of words: [word1, word2, ...]
    index = the index of the word we want to extract features for
    history = the list of predicted tags of the previous tokens
    """
    # Pad the sequence with placeholders
    # We will be looking at two words back and forward, so need to make sure we do not go out of bounds
    sentence = ['__START2__', '__START1__'] + list(sentence) + ['__END1__', '__END2__']
    
    # We will be looking two words back in history, so need to make sure we do not go out of bounds
    history = ['__START2__', '__START1__'] + list(history)
    
    # shift the index with 2, to accommodate the padding
    index += 2
    
    return {
            
                # Intrinsic features
                'word': sentence[index],
                'stem': stemmer.stem(sentence[index]),
                'shape': shape(sentence[index]),
                
                # Suffixes
                'suffix-1': sentence[index][-1],
                'suffix-2': sentence[index][-2:],
                'suffix-3': sentence[index][-3:],
                
                # Context
                'prev-word': sentence[index - 1],
                'prev-stem': stemmer.stem(sentence[index - 1]),
                'prev-prev-word': sentence[index - 2],
                'prev-prev-stem': stemmer.stem(sentence[index - 2]),
                'next-word': sentence[index + 1],
                'next-stem': stemmer.stem(sentence[index + 1]),
                'next-next-word': sentence[index + 2],
                'next-next-stem': stemmer.stem(sentence[index + 2]),
                
                # Historical features
                'prev-pos': history[-1],
                'prev-prev-pos': history[-2],
                
                # Composite
                'prev-word+word': sentence[index - 1].lower() + '+' + sentence[index],
                
            }
    
#Scikit Classifier Wrapper
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
class ScikitClassifier(nltk.ClassifierI):
    """
    Wrapper over a scikit-learn classifier
    """
    def __init__(self, classifier=None, vectorizer=None, model=None):
        if model is None:
            if vectorizer is None:
                vectorizer = DictVectorizer(sparse=False)
            
            if classifier is None:
                classifier = LogisticRegression()
            self.model = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
        else:
            self.model = model
                
    @property
    def vectorizer(self):
        return self.model[0][1]

    @property
    def classifier(self):
        return self.model[1][1]

    def train(self, featuresets, labels):
        self.model.fit(featuresets, labels)
        
    def partial_train(self, featuresets, labels, all_labels):
        self.model.partial_fit(featuresets, labels, all_labels)
        
    def test(self, featuresets, labels):
        self.model.score(featuresets, labels)
        
    def labels(self):
        return list(self.model.steps[1][1].classes_)
    
    def classify(self, featureset):
        return self.model.predict([featureset])[0]
    
    def classify_many(self, featuresets):
        return self.model.predict(featuresets)
    
#Tagger Training Function
from sklearn.svm import LinearSVC
#from classify import ScikitClassifier
def train_scikit_classifier(dataset):
    """
    dataset = list of tuples: [({feature1: value1, ...}, label), ...]
    """
    # split the dataset into featuresets and the predicted labels
    featuresets, labels = zip(*dataset)
    classifier = ScikitClassifier(classifier=LinearSVC())
    classifier.train(featuresets, labels)
    return classifier