import os
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from tool.trans import seq_translation


#               1  2 3 4 5  6 7  8 9  10 11 12 13 14 15 16
# orbbec pose   0  1 3 4 5  6 7  8 9  10 11 12 13 14 19 20
# kinectV2 pose 4 21 9 5 10 6 11 7 17 13 18 14 19 15 20 16
class MyData(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir        # dataset文件夹路径
        self.label = label
        root_path = os.path.join(self.root_dir, self.label)
        self.data_name_list = os.listdir(root_path)

    def __getitem__(self, idx):
        data_file = self.data_name_list[idx]
        datafile_path = os.path.join(self.root_dir, self.label, data_file)    # 数据样本路径
        body_data = read_skeleton(datafile_path)            # 获取一个样本数据存在二维数组中
        # preprocess
        body_data = seq_translation(body_data)
        max_val = 4.718282222747803
        min_val = -1.6540675908327103
        input_x = 255 * (body_data - min_val) / (max_val - min_val)
        rgb_ske = np.reshape(input_x, (input_x.shape[0], input_x.shape[1] // 3, 3))
        rgb_ske = resize(rgb_ske, output_shape=(224, 224)).astype(np.uint8)
        rgb_ske = np.transpose(rgb_ske, [1, 0, 2]).astype(np.float32)
        rgb_ske = np.transpose(rgb_ske, [2, 1, 0])  # 输入网络的伪图像
        # label = int(data_file[(data_file.find('A') + 1):(data_file.find('A') + 4)])    # label
        label = int(self.label)
        return rgb_ske, label

    def __len__(self):
        return len(self.data_name_list)


def read_skeleton(file):
    with open(file, 'r') as fr:
        str_data = fr.readlines()
    current_line = 0
    num_frames = int(str_data[current_line].strip('\r\n'))  # 每个文件的帧数

    bodies_data = np.zeros((num_frames, 48))

    for f in range(num_frames):
        current_line += 1
        num_bodies = int(str_data[current_line].strip('\r\n'))  # 第2行body数量,strip删除头尾
        for i in range(num_bodies):
            current_line += 1
            joints = []
            body_info = str_data[current_line]  # 第3行body id等信息，用不到

            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # 第4行num joints，但是我们只使用其中的16个关节点
            # 取出对应16个关节点
            # region
            # 第1个节点对应kinect第4个节点
            current_line += 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 第2个节点对应kinect第21个节点
            current_line = current_line + 17
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 3  9
            current_line = current_line - 12
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 4  5
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 5  10
            current_line = current_line + 5
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 6  6
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 7  11
            current_line = current_line + 5
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 8  7
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 9  17
            current_line = current_line + 10
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 10 13
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 11 18
            current_line = current_line + 5
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 12 14
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 13 19
            current_line = current_line + 5
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 14 15
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 15 20
            current_line = current_line + 5
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 16 16
            current_line = current_line - 4
            line_data = str_data[current_line].strip('\r\n').split()
            joints_one = np.array(line_data[:3], dtype=np.float32)
            joints = np.hstack((joints, joints_one))
            # 25
            current_line = current_line + 9
            # endregion

            bodies_data[f, :] = joints
    return bodies_data
