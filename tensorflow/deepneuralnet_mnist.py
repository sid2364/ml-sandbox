import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# number of nodes in the hidden layers
n_nodes_hidden_layer1 = 500
n_nodes_hidden_layer2 = 600
n_nodes_hidden_layer3 = 500

n_classes = 10 # ten digits
batch_size = 100 # run 100 samples and then tweak weights

# height * width
x = tf.placeholder("float", [None, 784]) # 784 is 28x28, since each handwritten digit is 28x28 pixels
y = tf.placeholder("float")

def neuralNetwork(data):
	hidden_1 = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hidden_layer1], stddev=0.1)), # passing the shape of the weight matrix
				'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hidden_layer1]))} # biases to add after weights are multiplied
	hidden_2 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hidden_layer1, n_nodes_hidden_layer2], stddev=0.1)),
				'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hidden_layer2]))}
	hidden_3 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hidden_layer2, n_nodes_hidden_layer3], stddev=0.1)),
				'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hidden_layer3]))}
	output_ = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hidden_layer3, n_classes], stddev=0.1)),
				'biases':tf.Variable(tf.constant(0.1,shape=[n_classes]))}

	# input * weights + biases
	l_1 = tf.add(tf.matmul(data, hidden_1['weights']), hidden_1['biases'])
	l_1 = tf.nn.relu(l_1) # rectified linear; activation function

	l_2 = tf.add(tf.matmul(l_1, hidden_2['weights']), hidden_2['biases'])
	l_2 = tf.nn.relu(l_2) # rectified linear; activation function

	l_3 = tf.add(tf.matmul(l_2, hidden_3['weights']), hidden_3['biases'])
	l_3 = tf.nn.relu(l_3) # rectified linear; activation function

	output = tf.matmul(l_3, output_['weights'] + output_['biases'])

	return output

def trainNeuralNetwork(x):
	prediction = neuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimiser = tf.train.AdamOptimizer().minimize(cost)

	epochs = 10

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				e_x, e_y = mnist.train.next_batch(batch_size)
				_, c = session.run([optimiser, cost], feed_dict={x:e_x, y:e_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) # assert, argmax returns index of max value, since this is 'one hot'
		print(prediction, y)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

trainNeuralNetwork(x)