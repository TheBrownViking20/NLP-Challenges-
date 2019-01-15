import numpy as np
import pandas as pd

import re
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import PassiveAggressiveClassifier

corpus = open('corpus.txt', 'r', encoding='utf-8-sig').read()

begin_title = '\n{3,}\s+THE SECRET CACHE\n{3,}.*'
corpus = re.search(begin_title, corpus, flags=re.M+re.S).group()
corpus = corpus.replace('\n', ' ') 
corpus = re.sub(r' {2,}', ' ', corpus)
corpus = corpus.replace('----', '')

valid_forms = ['am','are','were','was','is','been','being','be']
blank = '----'

tokens = wordpunct_tokenize(corpus)

def detect(tokens):
    return [t for t in tokens if t in valid_forms]
    
def replace_blank(tokens):
    return [blank if t in valid_forms else t for t in tokens]

def create_windows(tokens, window_size=5):
    X = []
    for i, word in enumerate(tokens):
        if word == blank:
            window = tokens[i-window_size:i] + tokens[i+1:i+window_size+1]
            window = ' '.join(window)
            X.append(window)    
    return X

targets = detect(tokens)
tokens = replace_blank(tokens)
X = create_windows(tokens)

l = LabelEncoder()
y = l.fit_transform(targets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
vectorizer = CountVectorizer()
classifier = PassiveAggressiveClassifier()
pipe = make_pipeline(vectorizer, classifier)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

import fileinput
f = fileinput.input()
N = int(f.readline())
sample = f.readline()
sample = wordpunct_tokenize(sample)
sample = create_windows(sample)
prediction = l.inverse_transform(pipe.predict(sample))

with open('output.txt', 'w') as f:
    f.write(prediction)
    print('\n'.join(prediction))
