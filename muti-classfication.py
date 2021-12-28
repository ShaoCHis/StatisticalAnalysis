from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from feature2 import getData
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import cycle

attr=['eye','mouth','head','mouth2','label']
data=getData()

X=data[attr[:4]]
#Y=label_binarize(data['label'], classes=[0, 1, 2])
Y=data['label']
print(X.shape)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.80,random_state=61)

tr_cls=DecisionTreeClassifier()
#tr_cls=SVC(4)
#tr_cls=GaussianNB()
#random_state = np.random.RandomState(0)
#tr_cls = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True))

tr_cls.fit(X_train,Y_train.astype('int'))
tr_cls.score(X_test,Y_test)

from sklearn.metrics import classification_report
prediction = tr_cls.predict(X_test)#
print(classification_report(Y_test,prediction))