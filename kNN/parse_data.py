import pandas as pd
import random

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
data = df.astype(float).values.tolist() # convert everything to float

random.shuffle(data)

test_size = 0.2

train_set = {2:[], 4:[]} # 2 is benign and 4 is malignant
test_set = {2:[], 4:[]} 

train_data = data[:-int(test_size*len(data))]
test_data = data[-int(test_size*len(data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])# i[-1] is last column

for i in test_data:
	test_set[i[-1]].append(i[:-1])

def get_data():
	return train_set, test_set
