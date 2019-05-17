# -- encoding:utf-8 --

"""
文件名：code_recognition
日期：Danycym
作者：2019/5/17
"""
import numpy as np
import tensorflow as tf
from verification_code_recognition import code_generate

code_char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'q', 'a', 'z', 'w', 's', 'x', 'e', 'd', 'c', 'r',
                 'f', 'v', 't', 'g', 'b', 'y', 'h', 'n', 'u', 'j',
                 'm', 'i', 'k', 'o', 'l', 'p', 'Q', 'A', 'Z', 'W',
                 'S', 'X', 'E', 'D', 'C', 'R', 'F', 'V', 'T', 'G',
                 'B', 'Y', 'H', 'N', 'U', 'J', 'M', 'I', 'K', 'O',
                 'L', 'P']
code_char_set_size = len(code_char_set)
code_char_2_number_dict = dict(zip(code_char_set, range(len(code_char_set))))
code_number_2_char_dict = dict(zip(range(len(code_char_set)), code_char_set))
# 网络中进行Dropout时候的，神经元的保留率(有多少神经元被保留下来)
# 0.75就表示75%的神经元保留下来，随机删除其中的25%的神经元(其实相当于将神经元的输出值设置为0)
keep_prob = 0.75
# 验证码中的字符数目
code_size = 4


def code_cnn(x, y):
    """
    构建一个验证码识别的CNN网络
    :param x:  Tensor对象，输入的特征矩阵信息，是一个4维的数据:[number_sample, height, weight, channels]
    :param y:  Tensor对象，输入的预测值信息，是一个2维的数据，其实就是验证码的值[number_sample, code_size]
    :return: 返回一个网络
    """
    """
    网络结构：构建一个简单的CNN网络，因为起始此时验证码图片是一个比较简单的数据，所以不需要使用那么复杂的网络结构，当然了：这种简单的网络结构，80%+的正确率是比较容易的，但是超过80%比较难
    conv -> relu6 -> max_pool -> conv -> relu6 -> max_pool -> dropout -> conv -> relu6 -> max_pool -> full connection -> full connection
    """
    # 获取输入数据的格式，[number_sample, height, weight, channels]
    x_shape = x.get_shape()
    # kernel_size_k: 其实就是卷积核的数目
    kernel_size_1 = 32
    kernel_size_2 = 64
    kernel_size_3 = 64
    unit_number_1 = 1024
    unit_number_2 = code_size * code_char_set_size

    with tf.variable_scope('net', initializer=tf.random_normal_initializer(0, 0.1), dtype=tf.float32):
        with tf.variable_scope('conv1'):
            w = tf.get_variable('w', shape=[5, 5, x_shape[3], kernel_size_1])
            b = tf.get_variable('b', shape=[kernel_size_1])
            net = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu1'):
            # relu6和relu的区别：relu6当输入的值大于6的时候，返回6，relu对于大于0的值不进行处理，relu6相对来讲具有一个边界
            # relu: max(0, net)
            # relu6: min(6, max(0, net))
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool1'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('conv2'):
            w = tf.get_variable('w', shape=[3, 3, kernel_size_1, kernel_size_2])
            b = tf.get_variable('b', shape=[kernel_size_2])
            net = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu2'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool2'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('dropout1'):
            tf.nn.dropout(net, keep_prob=keep_prob)
        with tf.variable_scope('conv3'):
            w = tf.get_variable('w', shape=[3, 3, kernel_size_2, kernel_size_3])
            b = tf.get_variable('b', shape=[kernel_size_3])
            net = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu3'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool3'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('fc1'):
            net_shape = net.get_shape()
            net_sample_feature_number = net_shape[1] * net_shape[2] * net_shape[3]
            net = tf.reshape(net, shape=[-1, net_sample_feature_number])
            w = tf.get_variable('w', shape=[net_sample_feature_number, unit_number_1])
            b = tf.get_variable('b', shape=[unit_number_1])
            net = tf.add(tf.matmul(net, w), b)
        with tf.variable_scope('softmax'):
            w = tf.get_variable('w', shape=[unit_number_1, unit_number_2])
            b = tf.get_variable('b', shape=[unit_number_2])
            net = tf.add(tf.matmul(net, w), b)
    return net


def text_2_vec(text):
    vec = np.zeros((code_size, code_char_set_size))
    k = 0
    for ch in text:
        index = code_char_2_number_dict[ch]
        vec[k][index] = 1
        k += 1
    return np.array(vec.flat)


def random_next_batch(batch_size=64, code_size=4):
    """
    随机获取下一个批次的数据
    :param batch_size:
    :param code_size:
    :return:
    """
    batch_x = []
    batch_y = []

    for i in range(batch_size):
        code, image = code_generate.generate_code_image(code_size)
        # code字符转换为数字的数组形式
        code_number = text_2_vec(code)
        batch_x.append(image)
        batch_y.append(code_number)

    return np.array(batch_x), np.array(batch_y)


def train_code_cnn(model_path):
    """
    模型训练
    :param model_path:
    :return:
    """
    # 1. 构建相关变量：占位符
    in_image_height = 60
    in_image_weight = 160
    x = tf.placeholder(tf.float32, shape=[None, in_image_height, in_image_weight, 1], name='x')
    y = tf.placeholder(tf.float32, shape=[None, code_size * code_char_set_size], name='y')
    # 2. 获取网络结构
    network = code_cnn(x, y)
    # 3. 构建损失函数（如果四个位置的值，只要有任意一个预测失败，那么我们损失就比较大）
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=y))
    # 4. 定义优化函数
    train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)
    # 5. 计算准确率
    predict = tf.reshape(network, [-1, code_size, code_char_set_size])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_y = tf.argmax(tf.reshape(y, [-1, code_size, code_char_set_size]), 2)
    correct = tf.equal(max_idx_p, max_idx_y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 6. 开始训练
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # a. 变量的初始化
        sess.run(tf.global_variables_initializer())

        # b. 开始训练
        step = 1
        while True:
            # 1. 获取批次的训练数据
            batch_x, batch_y = random_next_batch(batch_size=64, code_size=code_size)
            # 2. 对数据进行一下处理
            batch_x = tf.image.rgb_to_grayscale(batch_x)
            batch_x = tf.image.resize_images(batch_x, size=(in_image_height, in_image_weight),
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # 3. 训练
            _, cost_, accuracy_ = sess.run([train, cost, accuracy],
                                           feed_dict={x: batch_x.eval(), y: batch_y})
            print("Step:{}, Cost:{}, Accuracy:{}".format(step, cost_, accuracy_))

            # 4. 每10次输出一次信息
            if step % 10 == 0:
                test_batch_x, test_batch_y = random_next_batch(batch_size=64, code_size=code_size)
                # 2. 对数据进行一下处理
                test_batch_x = tf.image.rgb_to_grayscale(test_batch_x)
                test_batch_x = tf.image.resize_images(test_batch_x, size=(in_image_height, in_image_weight),
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                acc = sess.run(accuracy, feed_dict={x: test_batch_x.eval(), y: test_batch_y})
                print("测试集准确率:{}".format(acc))

                # 如果模型准确率0.7，模型保存，然后退出
                if acc > 0.7 and accuracy_ > 0.7:
                    saver.save(sess, model_path, global_step=step)
                    break

            step += 1
        saver.save(sess, model_path, global_step=step)
        # 模型可视化输出
        writer = tf.summary.FileWriter('./graph/code', tf.get_default_graph())
        writer.close()


if __name__ == '__main__':
    train_code_cnn('./model/code/capcha.model')
