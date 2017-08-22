'''
Data from: https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing, cross_validation

style.use("ggplot")

df = pd.read_excel('titanic.xls')

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
print(df.head())