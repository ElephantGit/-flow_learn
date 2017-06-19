import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# save mnist dataset in the folder of MNIST_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define variable weights and biases
def weight_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

# define convolution and pooling operation
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# define the shape of input and output and keep probability
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# reshape the input image
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# the first layer
w_conv1 = weight_variable([5,5,1,32]) # shape=[kernel, kernel, channel, featuremap]
b_conv1 = bias_variable([32]) # shape=[featuremap]
# convolution layer of layer1
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# pooling layer of layer1
h_pool1 = max_pool_2x2(h_conv1)

# the second layer
w_conv2 = weight_variable([5, 5, 32, 64]) # 64 is manual determine
b_conv2 = bias_variable([64])
# convolution layer of layer2
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# pooling layer of layer2
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64] reshape the h_pool2 from 3D tensor to 1D tensor
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # the output of second pooling layer is 7*7*64
w_fc1 = weight_variable([7*7*64, 1024])   # 1024 is manual determine
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# the last layer: output fully connected layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# loss function defined by cross entropy
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys) )

# train
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start the train
batch_size = 128
hm_epoches = 10
#for i in range(1000):
#    epoch_x, epoch_y = mnist.train.next_batch(1)
#    _, c = sess.run([train_step, cross_entropy], feed_dict={xs: epoch_x, ys: epoch_y, keep_prob: 0.5})
#    if i % 50 == 0:
#        print(c)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(hm_epoches):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_step, cross_entropy], feed_dict={xs: epoch_x, ys: epoch_y, keep_prob: 0.5})
            epoch_loss += c
        print('Epoch', epoch, 'completed out of', hm_epoches, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({xs: mnist.test.images, ys: mnist.test.labels}))
    save_path = saver.save(sess, "my_mnist_net.ckpt")
    print("Save to path: ", save_path)
