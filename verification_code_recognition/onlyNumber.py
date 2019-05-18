# -- encoding:utf-8 --

"""
文件名：onlyNumber
日期：Danycym
作者：2019/5/18
"""

import numpy as np
import random
# captcha是python验证码的库，安装方式pip install captcha
from captcha.image import ImageCaptcha
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

code_char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def random_captcha_text(char_set=code_char_set, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def generate_captcha_text_and_image():
    image = ImageCaptcha()
    code_text = random_captcha_text()
    code_text = ''.join(code_text)
    # 将字符串转换为验证码(流)
    captcha = image.generate(code_text)
    # 保存验证码图片
    # image.write(code_text, 'captcha/' + code_text + '.jpg')

    # 将验证码转换为图片的形式
    code_image = Image.open(captcha)
    code_image = np.array(code_image)

    return code_text, code_image


# 定义CNN网络
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WEIGHT, 1])
    # 3个卷积层
    # w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    w_c1 = tf.get_variable(name='w_c1', shape=[3, 3, 1, 32], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout1 = tf.nn.dropout(pool1, keep_prod)

    # w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    w_c2 = tf.get_variable(name='w_c2', shape=[3, 3, 32, 64], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout2 = tf.nn.dropout(pool2, keep_prod)

    # w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    w_c3 = tf.get_variable(name='w_c3', shape=[3, 3, 64, 128], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout3 = tf.nn.dropout(pool3, keep_prod)

    # 全连接
    # w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    w_d = tf.get_variable(name='w_d', shape=[8 * 20 * 128, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(dropout3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prod)

    # w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    w_out = tf.get_variable('w_out', shape=[1024, MAX_CAPTCHA * CHAR_SET_LEN], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WEIGHT])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60,160,3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = generate_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # 归一化
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


def train_crack_captcha_cnn():
    # 2. 获取网络结构
    network = crack_captcha_cnn()
    # 3. 构建损失函数（如果四个位置的值，只要有任意一个预测失败，那么我们损失就比较大）
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=Y))
    # 4. 定义优化函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    # 5. 计算准确率
    predict = tf.reshape(network, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_y = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct = tf.equal(max_idx_p, max_idx_y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prod: 0.75})
            print(step, loss_)

            # 每100step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prod: 1})
                print(step, acc)

                # 如果准确率大于85%，保存模型，完成训练
                if acc > 0.85:
                    saver.save(sess, './model/crack_captcha.model', global_step=step)
                    break
            step += 1


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/crack_captcha.model-1500')

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prod: 1})
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    train = 0
    if train == 0:
        text, image = generate_captcha_text_and_image()
        # print("验证码图像形状：", image.shape)  # (60, 160, 3)

        # 图像大小
        IMAGE_HEIGHT = 60
        IMAGE_WEIGHT = 160
        MAX_CAPTCHA = len(text)  # 验证码个数

        CHAR_SET_LEN = len(code_char_set)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WEIGHT])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prod = tf.placeholder(tf.float32)  # dropout

        train_crack_captcha_cnn()

    if train == 1:
        # 图像大小
        IMAGE_HEIGHT = 60
        IMAGE_WEIGHT = 160
        CHAR_SET_LEN = len(code_char_set)

        text, image = generate_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)
        plt.show()

        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten() / 255

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WEIGHT])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        keep_prod = tf.placeholder(tf.float32)  # dropout

        predict_text = crack_captcha(image)
        print("正确：{}，预测：{}".format(text, predict_text))
