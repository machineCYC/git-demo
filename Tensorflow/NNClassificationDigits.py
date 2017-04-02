import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data # (1797, 64)
y = digits.target # (1797,)
y = LabelBinarizer().fit_transform(y)
# 將y變成0,1的資料 ex:1, y = [0,1,0...,0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# define layer
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# define accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict = {xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict = {xs:v_xs, ys:v_ys})
    return result

# parameter
learning_rate = 0.5

# define placeholder for inputs
xs = tf.placeholder(tf.float32, [None, 64]) # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, activation_function = tf.nn.tanh)
prediction = add_layer(l1, 50, 10, activation_function = tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]))

# train step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    sess.run(train_step, feed_dict = {xs:X_train, ys:y_train})
    if i%50 == 0:
        print("cross_entropy = ", sess.run(cross_entropy, feed_dict = {xs: X_test, ys: y_test}),
              "accuracy = ", compute_accuracy(X_test, y_test))

# Predict the correct rate is 0.85 
