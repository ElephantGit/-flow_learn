import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


digits = load_digits()
x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
            # 在layer中爲weights和biases設置變化圖表
            tf.summary.histogram(layer_name + '/weights', weights)

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal([1, out_size]), name='b')
            # 在layer中爲weights和biases設置變化圖表
            tf.summary.histogram(layer_name + '/biases', biases)

        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
            wx_plus_b = tf.nn.dropout(wx_plus_b, 0.5)

        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

            # 在layer中爲weights和biases設置變化圖表
            tf.summary.histogram(layer_name + '/outputs', outputs)

        return outputs


# 保留概率，即我們要保留的結果所佔的比例，作爲一個palceholder，在run的時候傳入值，當=1時也就是100%保留，dropout沒有起作用
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 64], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 10], name='y_in')

# add hiden layer
l1 = add_layer(xs, 64, 50, n_layer='l1', activation_function=tf.nn.tanh)
# add output layer
prediction = add_layer(l1, 50, 10, n_layer='l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    # 設置cross_entropy的變化圖
    tf.summary.histogram('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()

# 給所有訓練圖合並
merged = tf.summary.merge_all()

# summary writer goes in here
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

for i in range(500):
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: x_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # plt.pause(0.1)
        print(sess.run(cross_entropy, feed_dict={xs: x_test, ys: y_test}))

