# -- encoding:utf-8 --

"""
文件名：08_device
日期：Danycym
作者：2019/5/6
"""

import tensorflow as tf

# with tf.device("/cpu:0"):
with tf.device("/cpu:0"):
    # 这个代码块中定义的操作，会在tf.device给定的设备上运行
    # 有一些操作，是不会再GPU上运行的（一定要注意）
    # 如果按照的tensorflow cpu版本，没法指定运行环境的
    a = tf.constant([1, 2, 3], name='a')
    b = tf.constant(2, name='b')
    c = tf.multiply(a, b)

# 新建Seesion，并将log_device_placement设置为True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 运行这个op
print(sess.run(c))
