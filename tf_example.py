import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
coefficient = np.array([[1.],[-20.],[100.]])

w= tf.Variable(0,dtype=tf.float32)
x = tf.placeholder(tf.float32,[3,1]) #data, not variables!!!
#cost = tf.add(tf.add(w**2,tf.multiply(-10.0,w)),25)
#cost = w**2-10*w+25
cost = x[0][0]*w**2+x[1][0]*w+x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer() #initialize variables to value determined
session = tf.Session() #Creates a tf session
session.run(init) #runs the variable
print(session.run(w))

for i in range(1000):
    session.run(train,feed_dict={x:coefficient})
print(session.run(w))