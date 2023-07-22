import os
import shutil
import random


def move_file(src_folder, to_folder, ratio):
	files = os.listdir(src_folder)
	num_files = len(files)
	num_files_to_move = int(num_files * ratio)
	files_to_move = random.sample(files, num_files_to_move)
	for file_name in files_to_move:
		src_path = os.path.join(src_folder, file_name)
		shutil.move(src_path, to_folder)


def move_file1(src_file, move_to_folder):
	if os.path.exists(src_file):
		shutil.move(src_file, move_to_folder)


def write_into_file(file_path):
	with open("./error1.txt", "a") as fr:
		fr.write(file_path+'\n')


def select_error_data(folder):
	file_list = os.listdir(folder)
	num = 1
	for file in file_list:
		print("{}/{}".format(num, len(file_list)))
		num = num + 1
		file_path = os.path.join(folder, file)
		with open(file_path, "r") as f:
			line_num = len(f.readlines())
			if (line_num - 1) % 28 != 0:
				write_into_file(file_path)


def delete_error_data(folder):
	file_list = os.listdir(folder)
	num = 1
	for file in file_list:
		print("{}/{}".format(num, len(file_list)))
		num = num + 1
		file_path = os.path.join(folder, file)
		with open(file_path, "r") as f:
			line_num = len(f.readlines())
		if (line_num - 1) % 28 != 0:
			os.remove(file_path)


if __name__ == '__main__':
	src_folder = "D:/HAR/dataset/NTU_dataset_mini/split/val/"
	for i in range(6):
		folder = os.path.join(src_folder, str(i))
		delete_error_data(folder)





