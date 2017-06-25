import tensorflow as tf
import cifar10

with tf.device('/cpu:0'):
    images, labels = cifar10.distorted_inputs()

xs = tf.placeholder('float', [128, 24, 24, 3])
ys = tf.placeholder('float', [128])

