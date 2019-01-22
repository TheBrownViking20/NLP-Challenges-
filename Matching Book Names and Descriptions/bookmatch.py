import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier

n=int(input())

train = input()
X_train = train.split("|")

y_train = np.arange(n)

clf = Pipeline([('tfidf-vect',TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), stop_words='english')),('tfidf',TfidfTransformer()),('clf',PassiveAggressiveClassifier())])

clf.fit(X_train, y_train)

test = input()
X_test = test.split("|")

predictions = clf.predict(X_test)

for i in predictions:
    print(i+1)
