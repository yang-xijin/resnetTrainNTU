# 坐标变换
import numpy as np


def seq_translation(data):
    origin = np.copy(data[:, 3:6])  # new origin: 第二个关节点 返回给定数组的深拷贝,不会随被拷贝值的改变而改变，浅拷贝(x=y)会改变
    data -= np.tile(origin, 16)  # 将origin先复制为16份，这一帧的数组减去它，即以它为原点进行坐标变换
    return data
