# -- encoding:utf-8 --

"""
文件名：10_variable_scope2
日期：Danycym
作者：2019/5/6
"""
import tensorflow as tf

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    with tf.variable_scope('foo', initializer=tf.constant_initializer(4.0)) as foo:
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(3.0))
        with tf.variable_scope('bar'):
            l = tf.get_variable("l", [1])

            with tf.variable_scope(foo):
                h = tf.get_variable('h', [1])
                g = v + w + l + h

    with tf.variable_scope('abc'):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(5.0))
        b = a + g

    sess.run(tf.global_variables_initializer())
    print("{},{}".format(v.name, v.eval()))
    print("{},{}".format(w.name, w.eval()))
    print("{},{}".format(l.name, l.eval()))
    print("{},{}".format(h.name, h.eval()))
    print("{},{}".format(g.name, g.eval()))
    print("{},{}".format(a.name, a.eval()))
    print("{},{}".format(b.name, b.eval()))

