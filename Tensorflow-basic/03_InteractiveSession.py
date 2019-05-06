# -- encoding:utf-8 --

"""
文件名：02_tensorflow
日期：Danycym
作者：2019/5/5
比较Session会话和InteractiveSession会话
"""
import tensorflow as tf

# 构建一个图
a = tf.constant(4)
b = tf.constant(3)
c = tf.multiply(a, b)

# 运行
with tf.Session():
    print(c.eval())

# 进入交互式会话
sess = tf.InteractiveSession()

# 定义变量和常量
x = tf.constant([1.0, 2.0])
a = tf.constant([2.0, 4.0])

# 进行减操作
sub = tf.subtract(x, a)

# 输出结果
print(sub.eval())
print(sess.run(sub))
