import tensorflow as tf
from tensorflow.contrib import rnn 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

epochs = 10
n_classes = 10 # ten digits
batch_size = 100 # run 100 samples and then tweak weights

chunk_size = 28
n_chunks = 28

rnn_size = 128

# height * width
x = tf.placeholder("float", [None, n_chunks, chunk_size]) # 784 is 28x28, since each handwritten digit is 28x28 pixels
y = tf.placeholder("float")

def recurrentNeuralNetwork(x):
	layer = {'weights':tf.Variable(tf.truncated_normal([rnn_size, n_classes], stddev=0.1)), # passing the shape of the weight matrix
		'biases':tf.Variable(tf.constant(0.1,shape=[n_classes]))} # biases to add after weights are multiplied

	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)

	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	
	output = tf.matmul(outputs[-1], layer['weights'] + layer['biases'])

	return output

def trainNeuralNetwork(x):
	prediction = recurrentNeuralNetwork(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimiser = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				e_x, e_y = mnist.train.next_batch(batch_size)
				e_x = e_x.reshape((batch_size, n_chunks, chunk_size))
				_, c = session.run([optimiser, cost], feed_dict={x:e_x, y:e_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) # assert, argmax returns index of max value, since this is 'one hot'
		print(prediction, y)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape(-1, n_chunks, chunk_size), y:mnist.test.labels}))

trainNeuralNetwork(x)
