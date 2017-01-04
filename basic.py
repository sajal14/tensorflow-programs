import tensorflow as tf


x1 = tf.constant(5)
x2 = tf.constant(6)


result = tf.mul(x1,x2)
#result is also the tensor
print(result)
#Till here only graph is built

#Create the session 
with tf.Session() as sess:
	out = sess.run(result)
	print(out)


print(out)
#We cant do sess.run(result) as, we are outside the session