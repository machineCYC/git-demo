### A simple neural network learning the Linear function
### ---
### 生成線性資料，並將資料增加一些雜訊，使它更像真實資料
### 在XY座標平面上呈現資料的分布狀況
### 更有系統性的建立簡單的神經網絡，並將loss function的結果打印出來
### ---

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
# parameter
learning_rate = 0.1

# generate data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # (300, 1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# input layer
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)

# output layer
prediction = add_layer(l1, 10, 1, activation_function = None)

# the loss between prediction and the true data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i %50 == 0:
        print(sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))

