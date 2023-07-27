import os
from PIL import Image
import imageio
import numpy as np
from skimage.transform import resize


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


def seq_translation(data):
    origin = np.copy(data[:, 3:6])  # new origin: 第二个关节点 返回给定数组的深拷贝,不会随被拷贝值的改变而改变，浅拷贝(x=y)会改变
    data -= np.tile(origin, 16)  # 将origin先复制为16份，这一帧的数组减去它，即以它为原点进行坐标变换
    return data


# 三通道图
def convert_img():
    labels = [0, 1, 2, 3, 4, 5]
    root_dir = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/"
    save_dir = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/images"

    for label in labels:
        filedir = os.path.join(root_dir, str(label))
        filelist = os.listdir(filedir)
        data_num = len(filelist)
        num = 0
        for filename in filelist:
            file_path = os.path.join(filedir, filename)
            body_data = read_skeleton(file_path)  # 获取一个样本数据存在二维数组中
            num = num + 1
            print("{}/{}".format(num, data_num))
            # preprocess
            body_data = seq_translation(body_data)
            max_val = 5.18858098984
            min_val = -5.28981208801
            input_x = 255 * (body_data - min_val) / (max_val - min_val)
            rgb_ske = np.reshape(input_x, (input_x.shape[0], input_x.shape[1] // 3, 3))
            rgb_ske = np.resize(rgb_ske, (224, 224, 3)).astype(np.float32)
            rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
            rgb_ske = np.add(rgb_ske, 0.5)      # 四舍五入
            rgb_ske = rgb_ske.astype(np.uint8)
            imgname = str(num) + 'A' + str(label) + ".bmp"
            imgpath = os.path.join(save_dir, imgname)
            img = Image.fromarray(rgb_ske)
            img.save(imgpath)

            print(img)


def conv_img(file, save_dir):
    body_data = read_skeleton(file)  # 获取一个样本数据存在二维数组中
    # preprocess
    body_data = seq_translation(body_data)
    max_val = 5.18858098984
    min_val = -5.28981208801
    input_x = 255 * (body_data - min_val) / (max_val - min_val)
    rgb_ske = np.reshape(input_x, (input_x.shape[0], input_x.shape[1] // 3, 3))
    rgb_ske = resize(rgb_ske, output_shape=(224, 224)).astype(np.float32)
    rgb_ske = np.transpose(rgb_ske, [1, 0, 2]).astype(np.uint8)
    imgname = "1.jpg"
    imgpath = os.path.join(save_dir, imgname)
    imageio.imsave(imgpath, rgb_ske)


def conv_tensor(file):
    body_data = read_skeleton(file)  # 获取一个样本数据存在二维数组中
    # preprocess
    body_data = seq_translation(body_data)
    max_val = 5.18858098984
    min_val = -5.28981208801
    input_x = 255 * (body_data - min_val) / (max_val - min_val)
    # rgb_ske = np.reshape(input_x, (input_x.shape[0], input_x.shape[1] // 3, 3))
    rgb_ske = resize(input_x, output_shape=(224, 224)).astype(np.int8)
    # rgb_ske = np.transpose(rgb_ske, [1, 0, 2])
    # rgb_ske = np.transpose(rgb_ske, [2, 1, 0])
    np.savetxt("./23.txt", rgb_ske)


# conv_jpg灰度图
def conv_jpg():
    labels = [0, 1, 2, 3, 4, 5]
    root_dir = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/"
    save_dir = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/images"

    for label in labels:
        filedir = os.path.join(root_dir, str(label))
        filelist = os.listdir(filedir)
        data_num = len(filelist)
        num = 0
        for filename in filelist:
            file_path = os.path.join(filedir, filename)
            body_data = read_skeleton(file_path)  # 获取一个样本数据存在二维数组中
            num = num + 1
            print("label{}:{}/{}".format(label, num, data_num))
            # preprocess
            body_data = seq_translation(body_data)
            max_val = 5.18858098984
            min_val = -5.28981208801
            input_x = 255 * (body_data - min_val) / (max_val - min_val)
            rgb_ske = resize(input_x, output_shape=(224, 224))
            # rgb_ske = np.reshape(rgb_ske, (1, rgb_ske.shape[0], rgb_ske.shape[1]))
            imgname = str(num) + 'A' + str(label) + ".jpg"
            imgpath = os.path.join(save_dir, imgname)
            img = Image.fromarray(rgb_ske)
            img = img.convert('L')
            img.save(imgpath)


if __name__ == "__main__":
    # file = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/2/S001C001P002R001A009.skeleton"
    # conv_tensor(file)
    convert_img()

