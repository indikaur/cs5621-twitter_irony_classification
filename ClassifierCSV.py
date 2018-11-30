from random import shuffle
import pandas as pd
import preprocessor as pp
import nltk
from nltk import TweetTokenizer
from nltk.classify.scikitlearn import SklearnClassifier


class ClassifierCSV:

    def __init__(self, csv_file, featureset_size=1000, test_ratio=0.1):
        self.csv_file = csv_file
        self.documents = []
        self.words = []
        self.featureset_size = featureset_size
        self.test_ratio = test_ratio
        self.feature_words = None
        self.classifier = None


    def _read_csv(self):
        data = pd.read_csv('train.txt', sep='\t', header=0)
        data.drop(['Tweet index'], axis=1, inplace=True)
        pp.set_options(pp.OPT.URL, pp.OPT.EMOJI, pp.OPT.MENTION)

        for i in range(len(data)):
            tt = TweetTokenizer()
            cleaned_test = pp.clean(data.iloc[i][1])
            tokens = tt.tokenize(cleaned_test)

            for t in tokens:
                if len(t) == 1 and t.isalpha() == False:
                    tokens.remove(t)

            doc, label = [w.lower() for w in tokens], data.iloc[i][0]

            for word in doc:
                self.words.append(word)

            self.documents.append((doc, label))


    def _generate_word_features(self):
        frequency_dist = nltk.FreqDist()

        for word in self.words:
            frequency_dist[word] += 1

        self.feature_words = list(frequency_dist)[:self.featureset_size]


    def __document_features(self, document):
        document_words = set(document)
        features = {}

        for word in self.feature_words:
            features['contains({})'.format(word)] = (word in document_words)

        return features


    def train_naive_bayes_classifier(self):
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()

        shuffle(self.documents)

        feature_sets = [(self.__document_features(d), c) for (d, c) in self.documents]
        cutoff = int(len(feature_sets) * self.test_ratio)
        train_set, test_set = feature_sets[cutoff:], feature_sets[:cutoff]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        print('Achieved {0:.2f}% accuracy against training set'.format(nltk.classify.accuracy(self.classifier, train_set)*100))
        print('Achieved {0:.2f}% accuracy against test set'.format(nltk.classify.accuracy(self.classifier, test_set)*100))


    def train_sklearn_classifier(self, sk_learn_classifier):
        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()

        shuffle(self.documents)

        feature_sets = [(self.__document_features(d), c) for (d, c) in self.documents]
        cutoff = int(len(feature_sets) * self.test_ratio)
        train_set, test_set = feature_sets[cutoff:], feature_sets[:cutoff]
        self.classifier = SklearnClassifier(sk_learn_classifier()).train(train_set)

        print('Achieved {0:.2f}% accuracy against training set'.format(nltk.classify.accuracy(self.classifier, train_set)*100))
        print('Achieved {0:.2f}% accuracy against test set'.format(nltk.classify.accuracy(self.classifier, test_set)*100))


    def classify_new_sentence(self, sentence):
        pp.set_options(pp.OPT.URL, pp.OPT.EMOJI, pp.OPT.MENTION)
        tt = tt = TweetTokenizer()
        cleaned_test = pp.clean(sentence)
        tokens = tt.tokenize(cleaned_test)

        for t in tokens:
            if len(t) == 1 and t.isalpha() == False:
                tokens.remove(t)

        doc = [w.lower() for w in tokens]

        if not self.feature_words:
            self._read_csv()
            self._generate_word_features()

        test_features = {}

        for word in self.feature_words:
            test_features['contains({})'.format(word.lower())] = (word.lower() in doc)

        return self.classifier.classify(test_features)
