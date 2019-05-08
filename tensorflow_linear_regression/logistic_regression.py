# -- encoding:utf-8 --

"""
文件名：softmax01
日期：Danycym
作者：2019/5/8
"""

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer, OneHotEncoder

# 1.模拟数据产生
np.random.seed(28)
n = 500
x_data = np.random.normal(loc=0, scale=2, size=(n, 2))  # 随机生成均值为0，标准差为2的500数据，2个类别
y_data = np.dot(x_data, np.array([[5], [3]]))
y_data = OneHotEncoder().fit_transform(Binarizer(threshold=0).fit_transform(y_data)).toarray()

# 构建最终画图的数据（数据点）
t1 = np.linspace(-8, 10, 100)
t2 = np.linspace(-8, 10, 100)
xv, yv = np.meshgrid(t1, t2)
x_test = np.dstack((xv.flat, yv.flat))[0]

plt.scatter(x_data[y_data[:, 0] == 0][:, 0], x_data[y_data[:, 0] == 0][:, 1], s=50, marker='+', c='red')
plt.scatter(x_data[y_data[:, 0] == 1][:, 0], x_data[y_data[:, 0] == 1][:, 1], s=50, marker='x', c='blue')
plt.show()

# 2.模型构建
# 构建数据输入占位符
# x/y:None的意思表示维度未知（那也就是我们可以传入任意的数据样本条数
# x:2表示变量的特征属性时2个特征，即输入样本的维度数目
# y:2表示是样本变量所属的类别的数目，类别是多少个，这里就是几
x = tf.placeholder(tf.float32, [None, 2], name='x')
y = tf.placeholder(tf.float32, [None, 2], name='y')

# 预测模型构建
# 构建权重w和偏置项b
# w第一个2是输入样本的维度数目
# w第二个2是样本的目标属性所属类别的数目（有多少类别，就为几个）
# b中的2是样本的目标属性所属类别的数目（有多少类别，就为几）
w = tf.Variable(tf.zeros([2, 2]), name='w')
b = tf.Variable(tf.zeros([2]), name='b')
# act(Tensor)是通过sigmoid函数转换后的一个概率值（矩阵形式）逻辑回归公式中
act = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 模型构建的损失函数
# tf.reduce_sum:求和，当参数为矩阵的时候，axis相当于1的时候，对每行求和（和Numpy API中的axis参数意义一样）
# tf.reduce_mean:求均值，当不给定任何axis参数的时候，表示求解全部所有数据的均值
cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(act), axis=1))

# 使用梯度下降求解
# 使用梯度下降，最小化误差
# learning_rate:要注意，不要过大，过大可能不收敛，也不要过小，过小收敛速度比较慢
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize((cost))

# 得到预测的类别是哪一个
# tf.argmax:对矩阵按行或列计算最大值对应的下标，和numpy中的一样
# tf.equal:是对比这两个居中或者向量的相等的元素，如果是相等的那就返回True，否则返回False
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 正确率（True转换为1，False转换为0）
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 总共训练迭代次数
training_epochs = 50
# 批次数量
num_batch = int(n / 10)
# 训练迭代次数（打印信息）
display_step = 5

with tf.Session() as sess:
    # 变量初始化
    sess.run(init)

    for epoch in range(training_epochs):
        # 模型训练
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # 对用于画图的数据进行预测
    # y_hat：是一个None*2的矩阵
    y_hat = sess.run(act, feed_dict={x: x_test})
    # 根据softmax分类的模型理论，获取每个样本对应出现概率最大的（值最大）
    # y_hat:是一个None*1的矩阵
    y_hat = np.argmax(y_hat, axis=1)

print("模型训练完成！")
# 画图展示一下
cm_light = mpl.colors.ListedColormap(['#bde1f5', '#f7cfc6'])
y_hat = y_hat.reshape(xv.shape)
plt.pcolormesh(xv, yv, y_hat, cmap=cm_light)  # 预测值
plt.scatter(x_data[y_data[:, 0] == 0][:, 0], x_data[y_data[:, 0] == 0][:, 1], s=50, marker='+', c='red')
plt.scatter(x_data[y_data[:, 0] == 1][:, 0], x_data[y_data[:, 0] == 1][:, 1], s=50, marker='o', c='blue')
plt.show()