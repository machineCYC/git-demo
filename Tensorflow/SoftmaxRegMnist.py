### Use softmax regression to classify mnist data
### ---
### 從 tensorflow.examples.tutorials 讀取 mnist data
### 打印出圖片最基本的資訊
### 將圖面上的每一個 pixel 經過 softmax regression，去辨識出最高機率的數字
### 利用訓練資料去訓練 sofrmax regression 的參數
### 將訓練好的參數對測試資料的 accuracy 打印出來，預測結果為0.91
### ---

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

# exploratory data shape
print(mnist.train.images.shape) # (55000, 784)
print(mnist.train.labels.shape) # (55000, 10)
print(mnist.test.images.shape) # (10000, 784)
print(mnist.test.labels.shape) # (10000, 10)

# print the 1st data
print(mnist.train.images[1,:]) # (784,)
print(mnist.train.labels[1,:]) # (10,)

# plot the 1st data
first_train_img = np.reshape(mnist.train.images[1, :], (28, 28))
plt.matshow(first_train_img, cmap = plt.get_cmap("gray"))
plt.show()

# parameters
learning_rate = 0.01
batch_size = 100 # 一次學習100個資料

# define placeholder for inputs to model
xs = tf.placeholder(tf.float32, [None, 784]) # None是指第一個維度可以是任意長度
ys = tf.placeholder(tf.float32, [None, 10])

Weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(xs, Weights) + biases)

# the error between prediction and real data
cross_entropy = -tf.reduce_sum(ys*tf.log(prediction))

# train step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict = {xs: batch_xs, ys: batch_ys})
    if i %50 == 0:
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #　tf.cast 將資料變成新的type
        print(sess.run(accuracy, feed_dict = {xs: mnist.test.images, ys: mnist.test.labels}))

# Predict the correct rate is 0.91
