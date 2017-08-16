from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
ys = np.array([3, 4, 7, 4, 5, 8, 6, 7, 9, 9], dtype=np.float64)

def get_best_fit_m_and_b(xs=xs, ys=ys):
	m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
		(mean(xs)**2 - mean(xs**2)))
	b = (mean(ys) - m * mean(xs))
	return m, b

def predict_y(m, b, predict_x):
	return m * predict_x + b

def squared_error(ys_original, ys_regression_line):
	return sum((ys_regression_line - ys_original)**2)

def coefficient_of_determination(ys_original, ys_regression_line):
	y_mean_line = [mean(ys_original)] * len(ys_original)
	squared_error_regression = squared_error(ys_original, ys_regression_line)
	squared_error_y_mean = squared_error(ys_original, y_mean_line)
	return 1 - (squared_error_regression / squared_error_y_mean)


m, b = get_best_fit_m_and_b(xs, ys)

regression_line_ys = [(m * x) + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line_ys)
print(r_squared)

#plt.scatter(xs, ys)
#plt.plot(xs, regression_line_ys)
#plt.show()
