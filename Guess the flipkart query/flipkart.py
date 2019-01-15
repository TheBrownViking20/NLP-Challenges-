from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import PassiveAggressiveClassifier as PAC
from sklearn.svm import LinearSVC as LSVC
from sklearn.ensemble import VotingClassifier as VC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
import re
import numpy as np

def get_words(query):
    return re.findall(r'(?:[a-zA-Z]+[a-zA-Z\'\-]?[a-zA-Z]|[a-zA-Z]+)',query)

X_train=[]
y_train=[]
count=0
with open('training.txt', 'r') as f:
    for sent in f:
        count+=1
        if count==1:
            continue
        a=[w for w in sent.rstrip().split("\t")]
        line=" ".join(word for word in get_words(a[0]))
        X_train.append(line)
        y_train.append(a[1])

X_train=np.array(X_train)
y_train=np.array(y_train)

mnb = MNB(alpha=0.01)
lr = LR(C=10,random_state=101)
svc = LSVC(C=10)
mlp = MLPClassifier()
pac = PAC()
knn = KNN(n_neighbors=3)

text_clf=Pipeline([('vect',CountVectorizer()),
                   ('tfidf',TfidfTransformer()),
                   ('clf', VC(estimators=[('lr', lr), ('svc', svc), ('pac', pac), ('mlp', mlp)], voting='hard'))])
text_clf.fit(X_train,y_train)

test=[]
for i in range(int(input())): 
    inp=input()
    line=" ".join(word for word in get_words(inp))
    test.append(inp)
y_pred=text_clf.predict(np.array(test))
for i in y_pred:
    print(i)
with open('output.txt', 'w') as f:
    f.writelines(y_pred)
