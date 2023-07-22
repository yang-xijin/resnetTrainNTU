import torch
from torch import nn

'''
    Block的各个plane值：
        inplane：输出block的之前的通道数
        midplane：在block中间处理的时候的通道数（这个值是输出维度的1/4）
        midplane*self.extention：输出的维度
'''


class Bottleneck(nn.Module):
    # 每个stage中维度拓展的倍数
    extention = 4

    # 定义初始化的网络和参数
    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplane, midplane, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(midplane)
        self.conv2 = nn.Conv2d(midplane, midplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(midplane)
        self.conv3 = nn.Conv2d(midplane, midplane * self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(midplane * self.extention)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 参差数据
        residual = x

        # 卷积操作
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # 是否直连（如果时Identity block就是直连；如果是Conv Block就需要对参差边进行卷积，改变通道数和size）
        if (self.downsample != None):
            residual = self.downsample(x)

        # 将参差部分和卷积部分相加
        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block, layers, num_classes):
        # self.inplane为当前的fm的通道数
        self.inplane = 64
        # CNN的卷积核通道数 = 卷积输入层的通道数 CNN的卷积输出层通道数(深度)= 卷积核的个数
        super(ResNet, self).__init__()

        # 参数
        self.block = block
        self.layers = layers

        # stem的网络层
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # 64，128，256，512是指扩大4倍之前的维度,即Identity Block的中间维度
        self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

        # 后续的网络
        self.avgpool = nn.AvgPool2d(7)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.extention, num_classes)

    def forward(self, x):

        # stem部分:conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def make_layer(self, block, midplane, block_num, stride=1):

        # block:block模块
        # midplane：每个模块中间运算的维度，一般等于输出维度/4
        # block_num：重复次数
        # stride：Conv Block的步长

        block_list = []

        # 先计算要不要加downsample模块
        downsample = None
        if (stride != 1 or self.inplane != midplane * block.extention):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, midplane * block.extention, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(midplane * block.extention)
            )

        # Conv Block
        conv_block = block(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = midplane * block.extention

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, midplane, stride=1))

        return nn.Sequential(*block_list)


'''
class ResidualBlock1(nn.Module):
    def __init__(self, channels_1, channels_2):
        super(ResidualBlock1, self).__init__()
        self.channels1 = channels_1
        self.channels2 = channels_2
        self.residual1 = nn.Sequential(
            nn.Conv2d(channels_1, channels_1, 1, 1),
            nn.BatchNorm2d(channels_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_1, channels_1, 3, 1, 1),
            nn.BatchNorm2d(channels_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_1, channels_2, 1),
            nn.BatchNorm2d(channels_2),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(channels_1, channels_2, 1),
            nn.BatchNorm2d(channels_2),
        )

    def forward(self, x):
        y1 = self.residual1(x)
        y2 = self.residual2(x)
        return nn.ReLU(y1 + y2)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.re1 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(3, 2, 1)
        self.res1 = ResidualBlock1(64, 256)
        self.conv2 = nn.Conv2d(256, 64, 1, 1)
        self.res2 = ResidualBlock1(64, 256)
        self.conv3 = nn.Conv2d(256, 64, 1, 1)
        self.res3 = ResidualBlock1(64, 256)
        self.conv4 = nn.Conv2d(256, 128, 1, 1)
        self.res4 = ResidualBlock1(128, 512)
        self.conv5 = nn.Conv2d(512, 128, 1, 1)
        self.res5 = ResidualBlock1(128, 512)
        self.conv6 = nn.Conv2d(512, 128, 1, 1)
        self.res6 = ResidualBlock1(128, 512)
        self.conv7 = nn.Conv2d(512, 128, 1, 1)
        self.res7 = ResidualBlock1(128, 512)
        self.conv8 = nn.Conv2d(512, 256, 1, 1)
        self.res8 = ResidualBlock1(256, 1024)
        self.conv9 = nn.Conv2d(1024, 256, 1, 1)
        self.res9 = ResidualBlock1(256, 1024)
        self.conv10 = nn.Conv2d(1924, 256, 1, 1)
        self.res10 = ResidualBlock1(256, 1024)
        self.conv11 = nn.Conv2d(1024, 256, 1, 1)
        self.res11 = ResidualBlock1(256, 1024)
        self.conv12 = nn.Conv2d(1024, 256, 1, 1)
        self.res12 = ResidualBlock1(256, 1024)
        self.conv13 = nn.Conv2d(1024, 256, 1, 1)
        self.res13 = ResidualBlock1(256, 1024)
        self.conv14 = nn.Conv2d(1024, 512, 1, 1)
        self.res14 = ResidualBlock1(512, 2048)
        self.conv15 = nn.Conv2d(1024, 512, 1, 1)
        self.res15 = ResidualBlock1(512, 2048)
        self.conv16 = nn.Conv2d(1024, 512, 1, 1)
        self.res16 = ResidualBlock1(512, 2048)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, 60)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.re1(x)
        x = self.max1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.res4(x)
        x = self.conv5(x)
        x = self.res5(x)
        x = self.conv6(x)
        x = self.res6(x)
        x = self.conv7(x)
        x = self.res7(x)
        x = self.conv8(x)
        x = self.res8(x)
        x = self.conv9(x)
        x = self.res9(x)
        x = self.conv10(x)
        x = self.res10(x)
        x = self.conv11(x)
        x = self.res11(x)
        x = self.conv12(x)
        x = self.res12(x)
        x = self.conv13(x)
        x = self.res13(x)
        x = self.conv14(x)
        x = self.res14(x)
        x = self.conv15(x)
        x = self.res15(x)
        x = self.conv16(x)
        x = self.res16(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

'''
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    resnet = ResNet()
    x = torch.randn(1, 3, 224, 224)
    x = resnet(x)
    print(x.shape)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
