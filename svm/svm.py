import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")


class SVM:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b'}
		if self.visualization:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1, 1, 1)

	def fit(self, data):
		self.data = data
		opt_dict = {}

		transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]

		data_ = []
		print(self.data)
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					data_.append(feature)

		print(data_)
		self.max_feature_value = max(data_)
		self.min_feature_value = min(data_)
		print(self.max_feature_value, self.min_feature_value)
		data_ = None

		step_sizes = [self.max_feature_value * 0.1,
				self.max_feature_value * 0.01,
				self.max_feature_value * 0.001]
		print(step_sizes)

		b_range_multiple = 5
		b_multiple = 5 # no need to take small steps like with 'w'

		latest_optima = self.max_feature_value * 10
		print(latest_optima)

		for step in step_sizes:
			w = np.array([latest_optima, latest_optima])
			print(w)
			optimized = False

			########################### CAN BE THREADED ############################ 
			while not optimized:
				for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
						self.max_feature_value * b_range_multiple, 
						step * b_multiple):
					for transformation in transforms:
						w_tr = w * transformation
						found_option = True
						for yi in self.data:
							for xi in self.data[yi]:
								if not yi * (np.dot(w_tr, xi) + b ) >= 1:
									found_option = False
			############################ CAN BE THREADED ############################ 
						if found_option:
							opt_dict[np.linalg.norm(w_tr)] = [w_tr, b]
				if w[0] < 0:
					optimized = True
				else:
					w = w - step
			norms = sorted([n for n in opt_dict])
			opt_choice = opt_dict[norms[0]]

			self.w = opt_choice[0]
			self.b = opt_choice[1]

			print("'w' and 'b' values: ", self.w, self.b, " with step size of ", step)

			latest_optima = opt_choice[0][0] + step * 2

	def predict(self, features):
		# sign of dot product of x and w + b
		classification = np.sign(np.dot
			(np.array(features), self.w) + self.b)
		if classification != 0 and self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
		return classification

	def visualize(self):
		[[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
		def hyperplane(x, w, b, v):
			# hyperplane = x.w + b = v
			return (-w[0]*x-b + v) / w[1]

		data_range = (self.min_feature_value * 0.9, self.max_feature_value*1.1)
		hyperplane_x_min = data_range[0]
		hyperplane_x_max = data_range[1]

		psv1 = hyperplane(hyperplane_x_min, self.w, self.b, 1)
		psv2 = hyperplane(hyperplane_x_max, self.w, self.b, 1)
		self.ax.plot([hyperplane_x_min,hyperplane_x_max], [psv1, psv2], 'b')

		nsv1 = hyperplane(hyperplane_x_min, self.w, self.b, -1)
                nsv2 = hyperplane(hyperplane_x_max, self.w, self.b, -1)
                self.ax.plot([hyperplane_x_min,hyperplane_x_max], [nsv1, nsv2], 'b')

		db1 = hyperplane(hyperplane_x_min, self.w, self.b, 0)
                db2 = hyperplane(hyperplane_x_max, self.w, self.b, 0)
                self.ax.plot([hyperplane_x_min,hyperplane_x_max], [db1, db2], 'k')

		plt.show()


data_dict = {-1: np.array([[1, 7],
				[2, 8],
				[3, 8],]),
		1: np.array([[5, 1],
				[6, -1],
				[7, 3],])}

test_data = [[0,10], [3,1], [-4, 5], [6, -5], [5,8], [1,4], [4,5], [6,7]]


svm = SVM()
svm.fit(data_dict)

for d in test_data:
	svm.predict(d)

svm.visualize()
