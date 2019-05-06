# -- encoding:utf-8 --

"""
文件名：04_Variables1
日期：Danycym
作者：2019/5/5
"""

import tensorflow as tf


# 创建一个变量，初始化值为变量3.0
a = tf.Variable(3.0)

# 创建一个常量
b = tf.constant(2.0)
c = tf.add(a, b)

# 启动图后，变量必须先进行初始化操作
# 增加一个初始化变量的op到图中
init_op = tf.initialize_all_variables()

# 启动图
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # 运行init_op
    sess.run(init_op)

    # 获取值
    print("a = {}".format(sess.run(a)))
    print("c = {}".format(c.eval))