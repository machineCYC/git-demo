### Use convolutional neural network to classify mnist data
### ---
### 從 tensorflow examples tutorials dataset 讀取 mnist data
### 建立第一層convolutional，利用 5x5的patch在圖片上做資料萃取，並將資料高度從 1 提升至 32
### 經由 max pooling 將圖片size由 28x28 降至 14x14
### 建立第二層convolutional，一樣利用 5x5的patch在圖片上做資料萃取，並將資料高度從 32 提升至 64
### 再一次經由 max pooling 將圖片size由 14x14 降至 7x7
### 建立第一層full connection，由1024個神經元所組成，並利用 dropout 避免 overfitting
### 建立第二層full connection，
### 將訓練好的模型運用在test data並將分類精準度打印出來，預測結果為0.96
### ---

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

# define accuracy
def compute_accuracy(v_xs, v_ys):
    global prediciton
    y_pre = sess.run(prediction, feed_dict = {xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs:v_xs, ys:v_ys, keep_prob:1})
    return result

# define weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# define biases
def biases_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# define convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")
    # strides = [1,a,b,1], a指的是水平方向的跨度，b則是垂直方向的跨度,
    # 頭尾兩個1是必須的

# define pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

# parameter
learning_rate = 1e-4
batch_size = 100

# define placeholder for inputs
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# -1 指的是不管xs樣本個數，指先考慮一個
# 1 指的是資料是黑白

# convolution1 layer
W_conv1 = weight_variable([5,5,1,32]) # patch5x5, patch起始高度1,目標是32
b_conv1 = biases_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # (28,28,32)
h_pool1 = max_pool_2x2(h_conv1) # (14,14,32)

# convolution2 layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = biases_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # (14,14,64)
h_pool2 = max_pool_2x2(h_conv2) # (7,7,64)

# fullconnection1 layer
W_fc1 = weight_variable([7*7*64, 1024]) # 1024個神經元
b_fc1 = biases_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# [n_samples,7,7,64]-->[n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fullconnection2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediciton and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices = [1]))

# train step
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict = {xs:batch_xs, ys:batch_ys, keep_prob: 0.5})
    if i %50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

