import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

input_size = 5
nb_pred_standing = 3
output=np.array(3)
input_feature=np.random.random_integers(size=(nb_pred_standing,input_size),low=0,high=2)
input_values = np.array([6,5,3])



values = tf.placeholder(tf.float32, [None,nb_pred_standing])
x_raw = tf.placeholder(tf.float32, [nb_pred_standing, input_size])
x_list = tf.split(x_raw, num_or_size_splits=nb_pred_standing,axis=0)

y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([1,input_size]))

x_list = tf.convert_to_tensor(x_list)
omega = tf.map_fn(fn = lambda x: tf.reduce_sum(tf.multiply(x,W)) ,elems=x_list)
q = tf.reduce_sum(tf.multiply(values,omega))

loss= tf.square(y - q)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for e in range(1000):
    sess.run(train_op, feed_dict={x_raw: input_feature, y: output.reshape(1,1), values: input_values.reshape(1,3)})
    pred = sess.run(q, feed_dict={x_raw: input_feature, values: input_values.reshape(1,3)})
    # loss = sess.run(loss, feed_dict={x_raw: input_feature,y: output.reshape(1,1), values: input_values.reshape(1,3)})
    print('epoch',e,'training loss ',pred, 'prediction', (pred-3)**2)

# sess.close()