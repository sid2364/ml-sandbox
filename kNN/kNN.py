'''
Data from: archive.ics.uci.edu/ml/datasets.html
http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
'''
import numpy as np
from collections import Counter
import parse_data


dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
new_features = [2,5]

def kNN(data, predict, k=3):
	if len(data) >= k:
		print('k is less than total voting groups')
	
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]
	return vote_result, confidence

train_set, test_set = parse_data.get_data()
total = correct = 0

for group in test_set:
	for data in test_set[group]:
		vote, confidence = kNN(train_set, data, k=5)
		if group == vote:
			correct += 1
		total += 1
print("Accuracy: ", correct*1.0/total)
