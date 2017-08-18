""" A simple neural network learning the Linear function
生成線性資料，並將資料增加一些雜訊，使它更像真實資料
在XY座標平面上呈現資料的分布狀況和估計拋物線
在XYZ座標平面上呈現Gradient descent 的過程
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate data
np.random.seed(seed = 1)
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = -2* np.square(x_data) - 5 + noise

# parameter
learning_rate = 0.8

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*np.square(x_data) + biases

# the loss between prediction and real data
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

W_list, b_list, loss_list= [], [], []
for i in range(201):
    W_, b_, loss_ = sess.run([Weights, biases, loss])
    W_list.append(W_); b_list.append(b_); loss_list.append(loss_)
    result, _ = sess.run([y, train])

# Visualiztion codes:
print("W_list=", W_list, "b_list=", b_list)
# 2D plot
plt.figure(1)
plt.scatter(x_data, y_data, c = "b")
plt.plot(x_data, result, "r-", lw = 2)
# 3D plot
y_pre = lambda W, b: W * x_data + b
fig = plt.figure(2); ax = Axes3D(fig)
W3D, b3D = np.meshgrid(np.linspace(-10, 6, 30), np.linspace(-14, 2, 30)) # parameter space
loss3D = np.array([np.mean(np.square(y_pre(W_, b_) - y_data)) for W_, b_ in zip(W3D.flatten(), b3D.flatten())]).reshape(W3D.shape)
ax.plot_surface(W3D, b3D, loss3D, rstride=1, cstride=1, cmap=plt.get_cmap("rainbow"), alpha=0.5)
ax.scatter(W_list[0], b_list[0], zs=loss_list[0], s=300, c="r")  # initial parameter place
ax.set_xlabel("W"); ax.set_ylabel("b")
ax.plot(W_list, b_list, zs=loss_list, zdir="z", c="r", lw=3) # plot 3D gradient descent
plt.show()
