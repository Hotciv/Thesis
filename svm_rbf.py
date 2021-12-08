from reading_datasets import *
import tensorflow as tf
import pandas as pd

tf.reset_default_graph()

#training hyper parameters

learning_rate = 0.000001
C = 20
gamma = 50

X = tf.placeholder(tf.float32, shape=(None,2))
Y = tf.placeholder(tf.float32, shape=(None,1))
landmark = tf.placeholder(tf.float32, shape=(None,2))

W = tf.Variable(np.random.random((num_data)),dtype=tf.float32)
B = tf.Variable(np.random.random((1)),dtype=tf.float32)

batch_size = tf.shape(X)[0]

#RBF Kernel
tile = tf.tile(X, (1,num_data))
diff = tf.reshape( tile, (-1, num_data, 2)) - landmark
tile_shape = tf.shape(diff)
sq_diff = tf.square(diff)
sq_dist = tf.reduce_sum(sq_diff, axis=2)
F = tf.exp(tf.negative(sq_dist * gamma))

WF = tf.reduce_sum(W * F, axis=1) + B

condition = tf.greater_equal(WF, 0)
H = tf.where(condition,  tf.ones_like(WF),tf.zeros_like(WF))

ERROR_LOSS = C * tf.reduce_sum(Y * tf.maximum(0.,1-WF) + (1-Y) * tf.maximum(0.,1+WF))
WEIGHT_LOSS = tf.reduce_sum(tf.square(W))/2

TOTAL_LOSS = ERROR_LOSS + WEIGHT_LOSS

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(TOTAL_LOSS)