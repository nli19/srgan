import argparse
import numpy as np
import sys
import os
sys.path.append('models/')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter 
import time

from dataloader import MyDataLoader
from SRGAN import Generator, Discriminator
from loss import GeneratorLoss

name = 'SRGAN'


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def getPSNRLoss():
  mseloss_fn = nn.MSELoss(reduction='none')

  def PSNRLoss(output, target):
    loss = mseloss_fn(output, target)
    loss = torch.mean(loss, dim=(1,2))
    loss = 10 * torch.log10(loss)
    mean = torch.mean(loss)
    return mean

  return PSNRLoss

psnr_func = getPSNRLoss()	

def test(netG, netD, test_loader, loss_func, epoch, device):
	netG.eval()
	test_psnr = 0.
	count = 0.
	time_list = []
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			
			torch.cuda.synchronize()
			start = time.time()
			output = netG(data)
			torch.cuda.synchronize()
			end = time.time()
			tmp_time = end - start
			time_list.append(tmp_time)
			print(tmp_time)
			
			psnr = psnr_func(output, target)
			test_psnr += psnr.item()*data.size()[0]
			count += data.size()[0]

	test_psnr /= count
	time_list = np.asarray(time_list)
	avg_time = np.mean(time_list) 
	print('Test Epoch: {} PSNR: {:.6f}'.format(epoch, test_psnr),flush=True)
  # print('Time: {:.4f}'.format(avg_time),flush=True)
	print('Time: {:.4f}'.format(avg_time),flush=True)
	return test_psnr

def main():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	# torch.manual_seed(0)
	# trainset = MyDataLoader('data', 'train', in_ch=3)
	testset = MyDataLoader('data', 'test', in_ch = 3)

	# train_loader = DataLoader(trainset, 8, shuffle = True)
	test_loader = DataLoader(testset, 1, shuffle = True)

	netG = Generator()
	print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	netD = Discriminator()
	print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

	netG = netG.to(device)
	netD = netD.to(device)
	netG = nn.DataParallel(netG)
	netD = nn.DataParallel(netD)
	loss_func = GeneratorLoss()

	checkpointD = torch.load('./checkpoints/'+str(name)+'_ModelD')
	checkpointG = torch.load('./checkpoints/'+str(name)+'_ModelG')

	netD.load_state_dict(checkpointD)
	netG.load_state_dict(checkpointG)


	test_loss = test(netG, netD, test_loader, loss_func, epoch, device)

if __name__ == '__main__':
  main()




