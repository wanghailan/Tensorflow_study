# -- encoding:utf-8 --

"""
文件名：04_Variables2
日期：Danycym
作者：2019/5/5
"""

import tensorflow as tf

# 创建一个变量
w1 = tf.Variable(tf.random_normal([10], stddev=0.5, dtype=tf.float32), name='w1')
# 基于第一个变量创建第二个变量
a = tf.constant(2, dtype=tf.float32)
w2 = tf.Variable(w1.initialized_value() * a, name='w2')

# 进行全局初始化
init_op = tf.initialize_all_variables()

# 启动图
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # 运行init_op
    sess.run(init_op)

    # 获取值
    #result = sess.run([w1, w2])
    result = sess.run(fetches=[w1, w2])
    print("w1 = {}\nw2 = {}".format(result[0], result[1]))
