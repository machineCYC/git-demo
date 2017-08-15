import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed
tf.set_random_seed(1)

# data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

# hyprtparameter
learning_rate = 0.001
batch_size = 128
train_iter = 100000

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10


# inputs
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Weights
weights = {
    "in":tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

    "out":tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

# Biases
biases = {
    "in":tf.Variable(tf.constant(0.1, shape = [n_hidden_units,])),

    "out":tf.Variable(tf.constant(0.1, shape = [n_classes, ]))
}

def RNN(X, weight, biases):

    X = tf.reshape(X, [-1, n_inputs])

    X_in = tf.matmul(X, weights["in"]) + biases["in"]

    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # Cell
    ################################################################
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    init_stat = lstm_cell.zero_state(batch_size, dtype = tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = init_stat, time_major = False)
    ################################################################
    results = tf.matmul(final_state[1], weights["out"]) + biases["out"]

    return results

prediction = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

step = 0
while step * batch_size < train_iter:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys})
    if step % 20 == 0:
        print("step = ", step, ",", "accuracy = ", sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys}))
    step += 1













