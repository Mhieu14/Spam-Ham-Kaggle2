import pandas as pd
import numpy as np
import os
import email
import email.policy
import re
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords


class EmailToWords(BaseEstimator, TransformerMixin):
    def __init__(self):        
        self.stemmer = nltk.PorterStemmer()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_to_words = []
        for text in X:
            if text is None:
                text = 'empty'
            text = text.lower()
            text = re.sub("[^a-zA-Z]+", " ", text)
            word_counts = Counter(text.split())

            stemmed_word_count = Counter()
            for word, count in word_counts.items():
                stemmed_word = self.stemmer.stem(word)
                stemmed_word_count[stemmed_word] += count
            word_counts = stemmed_word_count

            X_to_words.append(word_counts)
            stop_words = list(set(stopwords.words('english')))            
            for word in list(X_to_words):
                if word in stop_words:
                    del X_to_words[word]   
        return np.array(X_to_words)

from scipy.sparse import csr_matrix

class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += min(count, 10)
        self.most_common = total_word_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(self.most_common)}

        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word,0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))



from sklearn.pipeline import Pipeline

email_pipeline = Pipeline([
    ("Email to Words", EmailToWords()),
    ("Wordcount to Vector", WordCountToVector()),
])

#
class WordCountToVector_2(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += min(count, 10)
        self.most_common = total_word_count.most_common()[:self.vocabulary_size]
        self.vocabulary = {word: index + 1 for index, (word, count) in enumerate(self.most_common)}
        #print(self.most_common)
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
