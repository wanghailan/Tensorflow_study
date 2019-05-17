# -- encoding:utf-8 --

"""
文件名：model_restore
日期：Danycym
作者：2019/5/17
"""

import tensorflow as tf
from verification_code_recognition import code_recognition

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

# 1. 构建相关变量：占位符
in_image_height = 60
in_image_weight = 160
x = tf.placeholder(tf.float32, shape=[None, in_image_height, in_image_weight, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, code_size * code_char_set_size], name='y')


# 2、加载模型并预测
def predict_captcha(captcha_image, model_path):
    # a、构建网络
    output = code_recognition.code_cnn(x, y)

    saver = tf.train.Saver()
    # 开启会话
    with tf.Session() as sess:
        # 加载模型
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        predict = tf.argmax(tf.reshape(output, [-1, code_size, code_char_set_size]), 2)
        text_list = sess.run(predict, feed_dict={x: captcha_image.eval()})
        return text_list


# 3. 创建单个验证码数据，用于预测
batch_x, batch_y = code_recognition.random_next_batch(batch_size=1, code_size=code_size)

# 4.重塑哑编码形状为(4,code_char_set_size)
code_vec = batch_y[0].reshape((code_size, code_char_set_size))


# 5.将原本真实的标签数据（哑编码转为text）
def vec_2_text(vec):
    text = ''
    for char_vec in code_vec:
        for index, value in enumerate(char_vec):
            if value == 1.0:
                char = code_char_set[index]
                text += char
    return text


# 6. 对预测数据进行一下处理，由(1, 60, 160, 3)->(1, 60, 160, 1)
batch_x = tf.image.rgb_to_grayscale(batch_x)
code_img = tf.image.resize_images(batch_x, size=(in_image_height, in_image_weight),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# 7.模型路径
model_path = './model/code/'

# 8.模型预测
text_list = predict_captcha(code_img, model_path)


# 9.将模型预测结果转为text
def out_2_text(text_list):
    pre_text = ''
    for i in text_list[0]:
        char = code_char_set[i]
        pre_text += char
    return pre_text


print("真实值：{}，预测值：{}".format(vec_2_text(code_vec), out_2_text(text_list)))
