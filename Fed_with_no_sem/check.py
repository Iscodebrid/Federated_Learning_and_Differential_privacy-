'''
@Project ：Fed_with_no_sem 
@File    ：check.py
@IDE     ：PyCharm 
@Author  ：Yangjie
@Date    ：2024/5/9 22:25 
@清华源   ：-i https://pypi.tuna.tsinghua.edu.cn/simple
'''
import numpy as np

# 加载.npy文件
data = np.load("./data/Gowalla.npy", allow_pickle=True)

# 查看数组的形状
print("Array shape:", data.shape)

# 查看数组的内容
print("Array content:", data)



