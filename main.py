import numpy as np
from torchvision.transforms import transforms

from model.resnet50 import *
from datafeeder.img_feeder import MyData
from torch.utils.data import DataLoader
import os
from tool.train import train_val, plot_loss, plot_acc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./result'):
        os.makedirs('./result')

    # 参数设置
    num_classes = 6
    batch_size = 32
    learn_rate = 0.0001
    epoch = 100  # 训练轮数
    val_accuracy_all = 0
    # 数据集准备
    train_dir = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/images/train"
    val_dir = "D:/postgraduate/action-recognition/dataset/NTU_dataset_mini/images/val"
    train_data = MyData(train_dir,
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(
                                                          mean=[0.5049725, 0.46190482, 0.5082291],
                                                          std=[0.014596962, 0.040307738, 0.01857705])]))
    val_data = MyData(val_dir,
                      transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean=[0.5049725, 0.46190482, 0.5082291],
                                                        std=[0.014596962, 0.040307738, 0.01857705])])
                      )
    # 训练集大小
    train_data_size = len(train_data)
    # 测试集大小
    val_data_size = len(val_data)
    print("\ntrain on {}".format(train_data_size))
    print("val on {}".format(val_data_size))
    # dataloader加载
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size
                              , shuffle=True, drop_last=False, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size
                            , shuffle=True, drop_last=False, num_workers=4)
    # 创建网络模型
    model = ResNet(Bottleneck, [3, 4, 6, 3], 6)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # train val
    history = train_val(epoch, model, train_loader, train_data_size, val_loader, val_data_size, loss_fn, optimizer,
                        device)

    plot_loss(np.arange(0, epoch), history)
    plot_acc(np.arange(0, epoch), history)
