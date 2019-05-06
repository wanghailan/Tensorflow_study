# -- encoding:utf-8 --

"""
文件名：06_feed_placeholder
日期：Danycym
作者：2019/5/6
"""

import tensorflow as tf

# 创建占位符，创建图
m1 = tf.placeholder(tf.float32)
m2 = tf.placeholder(tf.float32)
m3 = tf.placeholder_with_default(4.0, shape=None)
output = tf.multiply(m1, m2)
ot1 = tf.add(m1, m3)

# 运行图
with tf.Session() as sess:
    print(sess.run(output, feed_dict={m1: 3, m2: 4}))
    print(output.eval(feed_dict={m1: 8, m2: 10}))
    print(sess.run(ot1, feed_dict={m1: 3, m3: 3}))
    print(sess.run(ot1, feed_dict={m1: 3}))

# 构建一个矩阵的乘法，但是矩阵在运行的时候给定
a = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='placeholder_1')
b = tf.placeholder(dtype=tf.float32, shape=[3, 2], name='placeholder_2')
c = tf.matmul(m1, m2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print("result:\n{}".format(
        sess.run(fetches=c, feed_dict={a: [[1, 2, 3], [4, 5, 6]], b: [[9, 8], [7, 6], [5, 4]]})))
    print("result:\n{}".format(c.eval(feed_dict={a: [[1, 2, 3], [4, 5, 6]], b: [[9, 8], [7, 6], [5, 4]]})))
