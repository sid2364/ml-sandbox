import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, y, test_X, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_X = test_X.reshape([-1, 28, 28, 1])

convNet = input_data(shape=[None, 28, 28, 1], name="input")

convNet = conv_2d(convNet, 32, 2, activation="relu")
#32 as size, 2 as window, activation function is rectified linear
convNet = max_pool_2d(convNet, 2)

convNet = conv_2d(convNet, 64, 2, activation="relu")
#64 as size, 2 as window, activation function is rectified linear
convNet = max_pool_2d(convNet, 2)

convNet = fully_connected(convNet, 1024, activation="relu")

convNet = dropout(convNet, 0.8)

convNet = fully_connected(convNet, 10, activation="softmax")

convNet = regression(convNet, optimizer="adam", learning_rate=0.01, loss="categorical_crossentropy", name="targets")

model = tflearn.DNN(convNet)

"""
model.fit(X, y,
	n_epoch=4,
	validation_set=(test_X, test_y),
	snapshot_step=500,
	show_metric=True,
	#run_id='mnist'
	)

model.save('tflearnconvnet.model')
"""
model.load('tflearnconvnet.model')

print(model.predict([test_X[1]]))
