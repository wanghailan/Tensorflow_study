# -- encoding:utf-8 --

"""
文件名：logistic_regression
日期：Danycym
作者：2019/5/8
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_linear_regression import input_data

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取数据
mnist = input_data.read_data_sets('data/', one_hot=True)

# 下载下来的数据集被分三个子集：
# 5.5W行的训练数据集（mnist.train），
# 5千行的验证数据集（mnist.validation)
# 1W行的测试数据集（mnist.test）。
# 因为每张图片为28x28的黑白图片，所以每行为784维的向量。
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

# 打印数据形状
print(trainimg.shape)
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)
print(trainlabel[0])

# 随机展示4张图片
nsample = 4
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    # reshape：格式变化
    curr_img = np.reshape(trainimg[i, :], (28, 28))  # 28 by 28 matrix
    # 获取最大值（10个数字中，只有一个为1，其它均为0，所以最大值极为数字对应的实际值）
    curr_label = np.argmax(trainlabel[i, :])  # Label
    # 矩阵图
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("第" + str(i) + "个图，实际数字为：" + str(curr_label))
    plt.show()

x = tf.placeholder("float", [None, 784])  # 784是维度，none表示的是无限多，程序自动识别
y = tf.placeholder("float", [None, 10])  # 10是输出维度，表示类别数字0-9
W = tf.Variable(tf.zeros([784, 10]))  # 每个数字是784像素点的，所以w与x相乘的话也要有784个
b = tf.Variable(tf.zeros([10]))  # ，10表示这个10分类的
# 这里只是简单初始化为0，可以以某种分布随机初始化

# 回归模型  w*x+b，然后再加上softmax，这里和逻辑回归中的公式相对应
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# cost function 均值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
# 优化
learning_rate = 0.01
# 使用梯度下降，最小化误差
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# 正确率
accr = tf.reduce_mean(tf.cast(pred, "float"))
# 初始化
init = tf.global_variables_initializer()

# 迭代次数
training_epochs = 50
# 批尺寸
batch_size = 100
# 每迭代5次显示一次结果
display_step = 5
# 开启会话
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):  # 遍历迭代次数
    avg_cost = 0.
    # 55000/100
    num_batch = int(mnist.train.num_examples / batch_size)  # 批次
    for i in range(num_batch):
        # 获取数据集 next_batch获取下一批的数据
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 模型训练
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
    # 满足5次的一个迭代
    if epoch % display_step == 0:
        feeds_train = {x: mnist.train.images, y: mnist.train.labels}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("批次: %03d/%03d 损失: %.9f 训练集准确率: %.3f 测试集准确率: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("训练完成")
