# -- encoding:utf-8 --

"""
文件名：02_tensorflow
日期：Danycym
作者：2019/5/5
"""

import tensorflow as tf

# 构建一个图
a = tf.constant('10', tf.string, name='a_const')
b = tf.string_to_number(a, out_type=tf.float64, name='str_2_double')
c = tf.to_double(5.0, name='to_double')
d = tf.add(b, c, name='add')

# 构建Session并执行图
# 1、构建GPU相关参数
gpu_options = tf.GPUOptions()
# per_process_gpu_memory_fraction:给定对于每一个进程，分配多少的GPU内存，默认为1
# 设置为0.5表示分配50%的GPU内存
gpu_options.per_process_gpu_memory_fraction = 0.5
# allow_growth：设置为True表示在GPU内存分配的时候，采用动态分配方式，默认为False
# 动态分配的意思是指，在启动之前，不分配全部的内存，根据需要后面动态的进行分配
# 在开启动态分配后，GPU内存部分自动释放，所以复杂、长时间运行的任务不建议开启
gpu_options.allow_growth = True

# 2、构建Graph优化的相关参数
optimizer = tf.OptimizerOptions(
    do_common_subexpression_elimination=True,  # 设置为True表示开启公共执行子句优化
    do_constant_folding=True,  # 设置为True表示开始常数折叠优化
    opt_level=0  # 设置为0,表示开启上述两个优化，默认为0
)

graph_options = tf.GraphOptions(optimizer_options=optimizer)

# 3、构建Session的Config相关参数
config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                              graph_options=graph_options, gpu_options=gpu_options)

# 4、构建Session并运行
with tf.Session(config=config_proto) as sess:
    print(sess.run(d))
