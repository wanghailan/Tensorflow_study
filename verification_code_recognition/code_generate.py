# -- encoding:utf-8 --

"""
文件名：code_generate
日期：Danycym
作者：2019/5/17
"""

import numpy as np
import random
# captcha是python验证码的库，安装方式pip install captcha
from captcha.image import ImageCaptcha
from PIL import Image
import matplotlib.pyplot as plt

code_char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'q', 'a', 'z', 'w', 's', 'x', 'e', 'd', 'c', 'r',
                 'f', 'v', 't', 'g', 'b', 'y', 'h', 'n', 'u', 'j',
                 'm', 'i', 'k', 'o', 'l', 'p', 'Q', 'A', 'Z', 'W',
                 'S', 'X', 'E', 'D', 'C', 'R', 'F', 'V', 'T', 'G',
                 'B', 'Y', 'H', 'N', 'U', 'J', 'M', 'I', 'K', 'O',
                 'L', 'P']

# 字符集长度
code_char_set_size = len(code_char_set)
# 字符-数字字典（字符转数字）
code_char_2_number_dict = dict(zip(code_char_set, range(len(code_char_set))))
# 数字-字符字典（数字转字符）
code_number_2_char_dict = dict(zip(range(len(code_char_set)), code_char_set))
# 验证码中的字符数目
code_size = 4


# 随机产生验证码的字符
def random_code_text(code_size=4):  # 传入验证码长度
    code_text = []
    for i in range(code_size):
        c = random.choice(code_char_set)  # 随机选择字符集中的一个字符
        code_text.append(c)  # 添加到列表中
    return code_text  # 返回一个长度为4的列表


# 加字符列表转换为一个验证码的Image对象
def generate_code_image(code_size=4):
    image = ImageCaptcha()
    code_text = random_code_text(code_size)
    code_text = ''.join(code_text)
    # 将字符串转换为验证码(流)
    captcha = image.generate(code_text)
    # 保存验证码图片
    # image.write(code_text, 'captcha/' + code_text + '.jpg')

    # 将验证码转换为图片的形式
    code_image = Image.open(captcha)
    code_image = np.array(code_image)

    return code_text, code_image


if __name__ == '__main__':
    text, image = generate_code_image(4)
    ax = plt.figure()
    ax.text(0.1, 0.9, text, ha='center', va='center')
    plt.imshow(image)
    plt.show()
