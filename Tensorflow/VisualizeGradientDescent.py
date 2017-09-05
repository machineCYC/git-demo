"""Visualize Gradient Descent
Target function: f(x)=sin(b*cos(a*x))
觀察參數a, b的參數空間，藉由3D圖可以發現有很多local minima
藉由3D圖可知，由不同的起始點會導致收斂到不同的local minima
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

REAL_PARAMS = [1.2, 2.5]
INIT_PARAMS = [[-1, 6],
               [5, 1],
               [2, 4.2]]

learning_rate = 0.01

# function
y_fun = lambda a, b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))

x = np.linspace(-1, 3, 200)
noise = np.random.rand(200)/10
# taeget value
y = y_fun(*REAL_PARAMS) + noise

# set a, b variable, loss function for each case
a1, b1 = [tf.Variable(initial_value=v, dtype=tf.float32) for v in INIT_PARAMS[0]]
pred1 = tf_y_fun(a1, b1)
loss1 = tf.reduce_mean(tf.square(y-pred1))
train_op1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)

a2, b2 = [tf.Variable(initial_value=v, dtype=tf.float32) for v in INIT_PARAMS[1]]
pred2 = tf_y_fun(a2, b2)
loss2 = tf.reduce_mean(tf.square(y-pred2))
train_op2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2)

a3, b3 = [tf.Variable(initial_value=v, dtype=tf.float32) for v in INIT_PARAMS[2]]
pred3 = tf_y_fun(a3, b3)
loss3 = tf.reduce_mean(tf.square(y-pred3))
train_op3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss3)

# Collect the parameter(a, b) and loss function when doing gradient descent
a_list1, b_list1, loss_list1 = [], [], []
a_list2, b_list2, loss_list2 = [], [], []
a_list3, b_list3, loss_list3 = [], [], []

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(400):
    a_1, b_1, loss_1 = sess.run([a1, b1, loss1])
    a_2, b_2, loss_2 = sess.run([a2, b2, loss2])
    a_3, b_3, loss_3 = sess.run([a3, b3, loss3])

    a_list1.append(a_1);b_list1.append(b_1);loss_list1.append(loss_1)
    a_list2.append(a_2);b_list2.append(b_2);loss_list2.append(loss_2)
    a_list3.append(a_3);b_list3.append(b_3);loss_list3.append(loss_3)

    sess.run(train_op1)
    sess.run(train_op2)
    sess.run(train_op3)

print("a1=", a_1, "b1=", b_1)
print("a2=", a_2, "b2=", b_2)
print("a3=", a_3, "b3=", b_3)

# visualization codes:
# 2D plot
plt.plot(x, y)
# 3D plot
fig = plt.figure(2)
ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2,7, 100), np.linspace(-2,7, 100))
cost3D = np.array([np.mean(np.square(y_fun(aa, bb)-y)) for aa, bb in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, cmap=plt.get_cmap("rainbow"), alpha=0.5)
# initial value for each case
ax.scatter(a_list1[0], b_list1[0], zs=loss_list1[0], s=300, c="red")
ax.scatter(a_list2[0], b_list2[0], zs=loss_list2[0], s=300, c="black")
ax.scatter(a_list3[0], b_list3[0], zs=loss_list3[0], s=300, c="blue")

# The track of gradient descent
ax.plot(a_list1, b_list1, zs=loss_list1, c="red")
ax.plot(a_list2, b_list2, zs=loss_list2, c="black")
ax.plot(a_list3, b_list3, zs=loss_list3, c="blue")

plt.show()
