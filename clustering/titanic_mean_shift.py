'''
Data from: https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import preprocessing
import kmeans

style.use("ggplot")

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

#import sklearn.utils
#df = sklearn.utils.shuffle(df)

df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
	cols = df.columns.values
	for col in cols:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]

		if df[col].dtype != np.int64 and df[col].dtype != np.float64:
			col_contents = df[col].values.tolist()
			unique_elements = set(col_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1
			df[col] = list(map(convert_to_int, df[col]))

	return df

df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

classifier = MeanShift()
classifier.fit(X)

labels = classifier.labels_
cluster_centers = classifier.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
	temp_df = original_df[(original_df['cluster_group'] == float(i))]
	survival_cluster = temp_df[(temp_df['survived'] == 1)]
	survival_rate = len(survival_cluster)*1.0/len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)

