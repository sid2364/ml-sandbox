import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs


X, y = make_blobs(n_samples=25, centers=3, n_features=2)

style.use("ggplot")

import numpy as np
#X = np.array([[2,1],[1,1],[6,8],[8,8],[1,1],[9,10],[4,2],[0,3],[1,9],[3,13],[2,12]])

#X = X*2
radius = 5
colors = 10* ["g", "r", "c", "b", "k"]

class MeanShift:
	def __init__(self, radius=None, radius_step=50):
		self.r = radius
		self.radius_step = radius_step

	def fit(self, data):
		if self.r is None:
			full_data_centroid = np.average(data, axis=0)
			full_data_norm = np.linalg.norm(full_data_centroid)
			self.r = full_data_norm*1.0 / self.radius_step
			print(self.r, full_data_norm, full_data_centroid)

		centroids = {}

		for i in range(len(data)):
			centroids[i] = data[i]
		

		weights = [i for i in range(self.radius_step)][::-1]

		while True:
			new_centroids = []
			for i in centroids:
				in_b = []
				centroid = centroids[i]
				weights = [i for i in range(self.radius_step)][::-1]

				for featureset in data:
					distance = np.linalg.norm(featureset - centroid)
					if distance == 0:
						distance += 0.0000000001
					weight_index = int(distance*1.0/self.r)
					if weight_index > self.radius_step - 1:
						weight_index = self.radius_step - 1
					to_add = (weights[weight_index]**2) * [featureset]
					in_b += to_add

				new_centroid = np.average(in_b, axis=0)
				new_centroids.append(tuple(new_centroid))
	
			uniques = sorted(list(set(new_centroids)))

			to_pop_from_list = []
			for i in uniques:
				for j in uniques:
					if i == j:
						pass
					elif np.linalg.norm(np.array(i) - np.array(j)) < self.r:
						# converge to same centroid!
						#to_pop_from_list.append(j)
						break
			for i in to_pop_from_list:
				try:
					uniques.remove(i)
				except ValueError:
					pass
					#if not in list, but WHY?
				
			previous_centroids = dict(centroids)
			centroids = {}
			
			for i in range(len(uniques)):
				centroids[i] = np.array(uniques[i])

			optimised = True

			for i in centroids:
				#sorting so it's easier to compare here
				if not np.array_equal(centroids[i], previous_centroids[i]):
					optimised = False
				if not optimised:
					break
			if optimised:
				break

			self.centroids = centroids
			print(centroids)

			self.classifications = {}
			
			for i in range(len(self.centroids)):
				self.classifications[i] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

	def predict(self, data):
		for featureset in data:
			distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)
			return classification


clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150)



plt.axes()
ax = plt.gca()
plt.scatter(X[:,0], X[:,1], s=100)

for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color="k", marker='*', s=150)
	c = plt.Circle((centroids[c][0], centroids[c][1]), radius=radius, fill=False, color='g')
	ax.add_patch(c)

plt.axis('scaled')
plt.show()
