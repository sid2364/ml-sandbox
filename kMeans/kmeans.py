import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use("ggplot")
data = np.array([[2,3], [3,1], [4,2], [6,8], [8,9],[10,5]])

class KMeans:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}
		for i in range(self.k):
			# randomly assign first k centroids
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			# is gonna change every time the centroids move
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []
				
			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)
			previous_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			
			optimised = True
			for c in self.centroids:
				original_centroid = previous_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
					optimised = False
			if optimised:
				break

	def predict(self, data):
		distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification



# # # # #

colors = ['r', 'b']

classifier = KMeans()
classifier.fit(data)

for centroid in classifier.centroids:
	plt.scatter(classifier.centroids[centroid][0], classifier.centroids[centroid][1], marker="o", color="k", s=100)

for classification in classifier.classifications:
	c = colors[classification]
	for featureset in classifier.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color=c, s=120)

#plt.show()

unknown_data = np.array([[1, 4], [7, 6], [4, 1], [3, 4], [9, 12]])

for unknown in unknown_data:
	classification = classifier.predict(unknown)
	plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=130)

plt.show()
