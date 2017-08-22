import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans


style.use("ggplot")
data = np.array([[2,3], [3,1], [4,2], [6,8], [8,9],[10,5]])
#plt.scatter(data[:,0], data[:,1])


classifier = KMeans(n_clusters=2)

classifier.fit(data)

labels, centroids = classifier.labels_, classifier.cluster_centers_
print(labels)
print(centroids)

colors = ['r.', 'b.']

for i in range(len(data)):
	plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:,0], centroids[:,1], marker='*')
plt.show()
