import os
import numpy as np
import torchvision
from skimage.transform import resize
from torch.utils.data import Dataset
from tools.trans import seq_translation
from PIL import Image


#               1  2 3 4 5  6 7  8 9  10 11 12 13 14 15 16
# orbbec pose   0  1 3 4 5  6 7  8 9  10 11 12 13 14 19 20
# kinectV2 pose 4 21 9 5 10 6 11 7 17 13 18 14 19 15 20 16
class MyData(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir        # dataset文件夹路径
		self.data_name_list = os.listdir(root_dir)

	def __getitem__(self, idx):
		data_file = self.data_name_list[idx]
		img_path = os.path.join(self.root_dir, data_file)
		img = Image.open(img_path)
		img_tensor = torchvision.transforms.ToTensor()(img)
		label = int(data_file[(data_file.find('A')+1)])
		return img_tensor, label

	def __len__(self):
		return len(self.data_name_list)

