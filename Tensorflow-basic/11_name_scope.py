# -- encoding:utf-8 --

"""
文件名：11_name_scope
日期：Danycym
作者：2019/5/6
"""

import tensorflow as tf

with tf.Session() as sess:
    with tf.name_scope('name1'):
        with tf.variable_scope('variable1'):
            v = tf.Variable(1.0, name='v')
            w = tf.get_variable(name='w', shape=[1], initializer=tf.constant_initializer(2.0))
            h = v + w

    with tf.variable_scope('variable2'):
        with tf.name_scope('name2'):
            v2 = tf.Variable(2.0, name='v2')
            w2 = tf.get_variable(name='w2', shape=[1], initializer=tf.constant_initializer(2.0))
            h2 = v2 + w2

    sess.run(tf.global_variables_initializer())
    print("{},{}".format(v.name, v.eval()))
    print("{},{}".format(w.name, w.eval()))
    print("{},{}".format(h.name, h.eval()))
    print("{},{}".format(v2.name, v2.eval()))
    print("{},{}".format(w2.name, w2.eval()))
    print("{},{}".format(h2.name, h2.eval()))