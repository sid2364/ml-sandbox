'''
Data from: archive.ics.uci.edu/ml/datasets.html
http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
'''
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(accuracy)


