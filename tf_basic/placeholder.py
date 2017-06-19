import tensorflow as tf
#tensorflow如果想要從外界傳入data， 那就需要用到tf.palceholder,然后再传入数据

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

#传值的工作是sess.run(***, feed_dict={input:[], **})完成
with tf.Session() as sess:
    print(sess.run(output, feed_dict = {input1: [7.], input2:[2.]}))

