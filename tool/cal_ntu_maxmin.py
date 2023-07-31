import os
import numpy as np
from datafeeder.data_read import read_skeleton
from tool.trans import seq_translation


def read_ske(data_file):
    body_data = read_skeleton(data_file)  # 获取一个样本数据存在二维数组中
    # preprocess
    body_data = seq_translation(body_data)
    max_val = body_data.max()
    min_val = body_data.min()
    return max_val, min_val


if __name__ == "__main__":
    root_path = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/split/val"
    label_dir = os.listdir(root_path)
    max_save = 0
    min_save = 0
    for label in label_dir:
        file_dir = os.path.join(root_path, label)
        file_list = os.listdir(file_dir)
        for file_name in file_list:
            data_file = os.path.join(file_dir, file_name)
            max1, min1 = read_ske(data_file)
            if max1 > max_save:
                max_save = max1
                print("max:{}\nmin:{}".format(max_save, min_save))
            if min1 < min_save:
                min_save = min1
                print("max:{}\nmin:{}".format(max_save, min_save))
    print("max:{}\nmin:{}".format(max_save, min_save))
