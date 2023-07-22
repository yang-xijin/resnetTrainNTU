import os

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model.resnet50 import *
from datafeeder.img_feeder import MyData
from torch.utils.data import DataLoader
import os
from model.resnet18 import ResNet18
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
	# 参数设置
	num_classes = 60
	batch_size = 32
	learn_rate = 0.0001
	epoch = 1000  # 训练轮数
	val_accuracy_all = 0
	# 数据集准备
	train_dir = "D:/HAR/dataset/NTU_dataset_mini/images/train"
	val_dir = "D:/HAR/dataset/NTU_dataset_mini/images/val"
	train_data = MyData(train_dir)
	val_data = MyData(val_dir)
	# train_data = MyData(train_dir, '0') + MyData(train_dir, '1') + MyData(train_dir, '2') +\
	# 			 MyData(train_dir, '3') + MyData(train_dir, '4') + MyData(train_dir, '5')
	# val_data = MyData(val_dir, '0') + MyData(val_dir, '1') + MyData(val_dir, '2') +\
	# 			 MyData(val_dir, '3') + MyData(val_dir, '4') + MyData(val_dir, '5')
	# 分割数据集

	train_data_size = len(train_data)
	val_data_size = len(val_data)
	print("\ntrain on {}".format(train_data_size))
	print("val on {}".format(val_data_size))

	# dataloader加载
	train_loader = DataLoader(dataset=train_data, batch_size=batch_size
							  , shuffle=True, drop_last=False, num_workers=3)
	val_loader = DataLoader(dataset=val_data, batch_size=batch_size
							, shuffle=True, drop_last=False, num_workers=3)

	# 创建网络模型
	# model = ResNet18()
	# model = model.cuda()
	model = ResNet(Bottleneck, [3, 6, 4, 3], 6)
	model = model.cuda()

	# 损失函数
	loss_fn = nn.CrossEntropyLoss().cuda()
	# 优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

	# 设置训练网络的一些参数
	total_train_step = 0  # 训练次数
	total_val_step = 0  # 验证次数
	val_accuracy = 0
	train_accuracy = 0
	best_accuracy = 0
	# 添加tensorboard
	# writer = SummaryWriter("./logs_train")

	for i in range(epoch):
		print("-------第{}轮训练开始-------".format(i + 1))

		# 训练
		model.train()
		total_train_loss = 0
		total_accuracy = 0
		for data in train_loader:
			body_rbg, targets = data
			body_rbg = body_rbg.cuda()
			targets = targets.cuda()
			outputs = model(body_rbg)

			# out = outputs.cpu()
			# print(targets)
			# print(out)

			loss = loss_fn(outputs, targets)
			# 优化器调优
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			accuracy = (outputs.argmax(1) == targets).sum()

			total_accuracy = total_accuracy + accuracy
			total_train_loss += loss
			total_train_step += 1
			if total_train_step % 30 == 0:
				print("训练次数：{}, loss：{}".format(total_train_step, loss))
			# writer.add_scalar("train_loss", loss, total_train_step)
		print("整体训练集上的loss：{}, 整体训练集上的accu：{}".format(total_train_loss, total_accuracy / train_data_size))
		if i % 100 == 0 or i == epoch - 1:
			torch.save(model, "./results/best{}.pt".format(i))
			print("model saved")

		# 测试步骤
		model.eval()
		total_test_loss = 0
		total_accuracy = 0
		with torch.no_grad():
			for data in val_loader:
				body_rbg, targets = data
				body_rbg = body_rbg.cuda()
				targets = targets.cuda()
				outputs = model(body_rbg)
				accuracy = (outputs.argmax(1) == targets).sum()
				total_accuracy = total_accuracy + accuracy

				with torch.no_grad():
					loss = loss_fn(outputs, targets)
				total_test_loss += loss
		print("整体测试集上的loss：{}, 整体测试集上的accu：{}".format(total_test_loss, total_accuracy / val_data_size))

		total_val_step += 1

	# writer.close()
