import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''

Feedforward : input ==> Weights ==> HiddenLayer1 (Activation Fuction ) ==>.....
..... ==> Output layer 

Cross-entropy to check the cost/loss.
Optimization : minimize the cost by travelling backwards..E.g. AdamOptimizer,SGD,AdaGrad (Backpropagation)

epoch = feedforward + Backpropagation 

'''

#Only one component is on, rest is off
#Used for multiclass classification
# 0 = [1,0,0,0,0,0,0,0,0,0,0], 1= [0,1,0,0,0,0,0,0,0,0]....
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True) 
#X = 28 * 28 images

#3 hidden layers with 500 units each
num_nodes_h1 = 500
num_nodes_h2 = 500
num_nodes_h3 = 500


num_classes = 10
batch_size = 100 #Only 100 instances at a time because of memory constraint

#height * width
x = tf.placeholder('float',[None, 784]) #Flatten the image 28*28 = 784
y = tf.placeholder('float')
#Tensorflow will throw an error afterwards if the shape is not matched

#Creates the computation graph
def nn_model(data):
	# h1_out = input * weights + bias (Evem if all i/p is zero, still the neuron fires because of bias)
	h1_layer = {'weights' : tf.Variable(tf.random_normal([784,num_nodes_h1])),
	'biases' : tf.Variable(tf.random_normal([num_nodes_h1]))}
	h2_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_h1,num_nodes_h2])),
	'biases' : tf.Variable(tf.random_normal([num_nodes_h2]))}
	h3_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_h2,num_nodes_h3])),
	'biases' : tf.Variable(tf.random_normal([num_nodes_h3]))}
	output_layer = {'weights' : tf.Variable(tf.random_normal([num_nodes_h3,num_classes])),
	'biases' : tf.Variable(tf.random_normal([num_classes]))}

	l1 = tf.add(tf.matmul(data,h1_layer["weights"]) , h1_layer["biases"] )
	l1 = tf.nn.relu(l1) #Actication function is rectified linear unit

	l2 = tf.add(tf.matmul(l1,h2_layer["weights"]) , h2_layer["biases"] )
	l2 = tf.nn.relu(l2) #Actication function is rectified linear unit

	l3 = tf.add(tf.matmul(l2,h3_layer["weights"]) , h3_layer["biases"] )
	l3 = tf.nn.relu(l3) #Actication function is rectified linear unit

	output = tf.add(tf.matmul(l3,output_layer["weights"]) , output_layer["biases"] )

	return output


def train_nn(x):
	prediction = nn_model(x)
	#Calculate the difference b/w prediction and known labels
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#default learning rate = 0.01
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost],feed_dict= {x:epoch_x, y:epoch_y} ) #Run the optimizer with the cost and put the feed_dict
				epoch_loss += c
			print('# Epoch', epoch, 'loss :', epoch_loss)
	#Training ends here


		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print("Accuracy :" , accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_nn(x)
#95% accuracy


