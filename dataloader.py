import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image


class MyDataLoader(Dataset):
	def __init__(self, data_dir, partition, in_ch=3):
		self.in_ch = in_ch
		self.partition = partition
		hr = data_dir+'/HR/'
		lr = data_dir + '/LR/'
		self.hr_file = os.path.join(hr, '*.png')
		self.lr_file = os.path.join(lr, '*.png')
		self.hr_list = sorted(glob(self.hr_file))
		self.lr_list = sorted(glob(self.lr_file))
	
		# input 510 x 339 -> 2040 x 1356
		self.hr = []
		self.lr = []
		hr_patch_size = 204
		lr_patch_size = 51
		if self.partition == 'train':
			for f in self.hr_list:
				img = Image.open(f)
				img = img.resize((2040, 1356),resample=Image.BICUBIC)
				img.load()
				data = np.asarray(img, dtype='int32')
				data = data/255.0
				data = data.transpose(2, 1, 0)
				for i in range(data.shape[1]//hr_patch_size):
					for j in range(data.shape[2]//hr_patch_size):
						patch = data[:,hr_patch_size*i:hr_patch_size*(i+1),hr_patch_size*j:hr_patch_size*(j+1)]
						self.hr.append(patch)

			self.hr = np.asarray(self.hr, dtype=np.float32)

			for f in self.lr_list:
				img = Image.open(f)
				img = img.resize((510, 339),resample=Image.BICUBIC)
				img.load()
				data = np.asarray(img, dtype='int32')
				data = data/255.0
				data = data.transpose(2, 1, 0)
				for i in range(data.shape[1]//lr_patch_size):
					for j in range(data.shape[2]//lr_patch_size):
						patch = data[:,lr_patch_size*i:lr_patch_size*(i+1),lr_patch_size*j:lr_patch_size*(j+1)]
						self.lr.append(patch)
			self.lr = np.asarray(self.lr, dtype=np.float32)

		else:
			for f in self.hr_list:
				img = Image.open(f)
				img = img.resize((2040, 1356),resample=Image.BICUBIC)
				img.load()
				data = np.asarray(img, dtype='int32')
				data = data/255.0
				data = data.transpose(2, 1, 0)
				# print(data.shape)
				# data = np.resize(data, (3, 2040, 1356))
				self.hr.append(data)
			self.hr = np.asarray(self.hr, dtype=np.float32)

			for f in self.lr_list:
				img = Image.open(f)
				img = img.resize((510, 339),resample=Image.BICUBIC)
				img.load()
				data = np.asarray(img, dtype='int32')
				data = data/255.0
				data = data.transpose(2, 1, 0)
				# data = np.resize(data, (3, 510, 339))
				# data = np.resize(data, (3, 2040, 1356))
				self.lr.append(data)
			self.lr = np.asarray(self.lr,dtype=np.float32)



	def __len__(self):
		return len(self.hr)

	def __getitem__(self,idx):
		lr = self.lr[idx]
		hr = self.hr[idx]
		return lr, hr

###test

# testset = MyDataLoader('data/', 'test', in_ch = 3)
# lr,hr = testset.__getitem__(1)
# print(lr.shape)
# print(hr.shape)
		