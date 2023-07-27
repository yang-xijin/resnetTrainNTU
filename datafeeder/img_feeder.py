import os
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# normMean = [0.5049725, 0.46190482, 0.5082291]
# normStd = [0.014596962, 0.040307738, 0.01857705]
#               1  2 3 4 5  6 7  8 9  10 11 12 13 14 15 16
# orbbec pose   0  1 3 4 5  6 7  8 9  10 11 12 13 14 19 20
# kinectV2 pose 4 21 9 5 10 6 11 7 17 13 18 14 19 15 20 16
class MyData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # dataset文件夹路径
        self.data_name_list = os.listdir(root_dir)
        self.transform = transform

    def __getitem__(self, idx):
        data_file = self.data_name_list[idx]
        img_path = os.path.join(self.root_dir, data_file)
        img = Image.open(img_path).convert('RGB')
        img_array = np.asarray(img, dtype=np.float32)

        label = int(data_file[(data_file.find('A') + 1)])
        if self.transform is not None:
            img_array = self.transform(img_array)
        # img_array = np.transpose(img_array, [2, 0, 1])
        return img_array, label

    def __len__(self):
        return len(self.data_name_list)
