import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate data
np.random.seed(seed = 1)
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 2D plot
plt.plot(x_data, y_data)
plt.show()

# parameter
learning_rate = 0.5

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

for i in range(201):
    sess.run(train)
    if i%20 == 0:
        print(i, sess.run(Weights), sess.run(biases))
