import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell


hm_epochs = 3
num_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128 #512

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) 


#height * width
x = tf.placeholder('float',[None, n_chunks,chunk_size]) #Flatten the image 28*28 = 784
y = tf.placeholder('float')
#Tensorflow will throw an error afterwards if the shape is not matched

#Creates the computation graph
def rnn_model(data):
	layer = {'weights' : tf.Variable(tf.random_normal([rnn_size,num_classes])),
	'biases' : tf.Variable(tf.random_normal([num_classes]))}
	global x
	x = tf.transpose(x,[1,0,2])
	x = tf.reshape(x,[-1,chunk_size])
	x = tf.split(0,n_chunks,x)

	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)

	outputs, states = rnn.rnn(lstm_cell,x,dtype=tf.float32)

	output = tf.matmul(outputs[-1],layer["weights"])+layer["biases"]

	return output


def train_nn(x):
	prediction = rnn_model(x)
	#Calculate the difference b/w prediction and known labels
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#default learning rate = 0.01
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
				# epoch_x = epoch_x.resphape((batch_size,n_chunks,chunk_size))
				_, c = sess.run([optimizer,cost],feed_dict= {x:epoch_x, y:epoch_y} ) #Run the optimizer with the cost and put the feed_dict
				epoch_loss += c
			print('# Epoch', epoch, 'loss :', epoch_loss)
	#Training ends here


		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print("Accuracy :" , accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}))


train_nn(x)
#95% accuracy


