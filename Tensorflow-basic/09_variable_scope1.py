# -- encoding:utf-8 --

"""
文件名：09_variable_scope1
日期：Danycym
作者：2019/5/6
"""
import tensorflow as tf


# 方式一：不加作用域的
def my_func1(x):
    w1 = tf.Variable(tf.random_normal([1]))[0]
    b1 = tf.Variable(tf.random_normal([1]))[0]
    result1 = w1 * x + b1

    w2 = tf.Variable(tf.random_normal([1]))[0]
    b2 = tf.Variable(tf.random_normal([1]))[0]
    result2 = w2 * x + b2

    return result1, w1, b1, result2, w2, b2


# 下面两行代码还是属于图的构建
x = tf.constant(3, dtype=tf.float32)
r = my_func1(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 执行结果
    print(sess.run(r))


# 方式二：加作用域
def my_func2(x):
    # initializer：初始化器
    w = tf.get_variable('weight', [1], initializer=tf.random_normal_initializer())[0]
    b = tf.get_variable('bias', [1], initializer=tf.random_normal_initializer())[0]
    result = w * x + b

    return result, w, b


def func(x):
    with tf.variable_scope('op1', reuse=tf.AUTO_REUSE):
        r1 = my_func2(x)
    with tf.variable_scope('op2', reuse=tf.AUTO_REUSE):
        r2 = my_func2(x)
    return r1, r2


# 下面两行代码还是属于图的构建
x1 = tf.constant(3, dtype=tf.float32, name='x1')
x2 = tf.constant(4, dtype=tf.float32, name='x2')
with tf.variable_scope('func1'):  # 支持嵌套
    r1 = func(x1)
with tf.variable_scope('func2'):
    r2 = func(x2)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 执行结果
    print(sess.run([r1, r2]))
