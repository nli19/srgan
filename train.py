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
from torch.autograd import Variable

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

def train(netG, netD, train_loader, optimizerG, optimizerD, loss_func, epoch, device):
	netG.train()
	netD.train()

	running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
	for batch_idx, (data, target) in enumerate(train_loader):
		batch_size = data.size(0)
		running_results['batch_sizes'] += batch_size
		data, target = data.to(device), target.to(device)
		g_update_first = True

		real_image = Variable(target)
		real_image = real_image.to(device)
		z = Variable(data)
		z = z.to(device)
		fake_img = netG(z)

		netD.zero_grad()
		real_out = netD(real_image).mean()
		fake_out = netD(fake_img).mean()
		d_loss = 1 - real_out + fake_out
		d_loss.backward(retain_graph=True)

		netG.zero_grad()
		g_loss = loss_func(fake_out, fake_img, real_image)
		g_loss.backward(retain_graph=True)

		fake_img = netG(z)
		fake_out = netD(fake_img).mean()
		
		optimizerD.step()
		optimizerG.step()

		running_results['g_loss'] += g_loss.item() * batch_size
		running_results['d_loss'] += d_loss.item() * batch_size
		running_results['d_score'] += real_out.item() * batch_size
		running_results['g_score'] += fake_out.item() * batch_size

		print('[%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

		return running_results['d_loss']
		

def test(netG, netD, test_loader, loss_func, epoch, device):
	netG.eval()
	test_psnr = 0.
	count = 0.
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			output = netG(data)
			psnr = psnr_func(output, target)
			test_psnr += psnr.item()*data.size()[0]
			count += data.size()[0]
	test_psnr /= count
	print('Test Epoch: {} PSNR: {:.6f}'.format(epoch, test_psnr),flush=True)
  # print('Time: {:.4f}'.format(avg_time),flush=True)
	return test_psnr

def main():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	torch.manual_seed(0)
	trainset = MyDataLoader('data', 'train', in_ch=3)
	testset = MyDataLoader('data', 'test', in_ch = 3)

	train_loader = DataLoader(trainset, 8, shuffle = True)
	test_loader = DataLoader(testset, 8, shuffle = True)

	netG = Generator()
	print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
	netD = Discriminator()
	print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

	netG = netG.to(device)
	netD = netD.to(device)
	netG = nn.DataParallel(netG)
	netD = nn.DataParallel(netD)
	loss_func = GeneratorLoss()


	optimizerG = optim.Adam(netG.parameters())
	optimizerD = optim.Adam(netD.parameters())

	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.9)
	start_epoch = 0
	best_loss = 100000

	for epoch in range(start_epoch+1, start_epoch+2+1):
		train_loss = train(netG, netD, train_loader, optimizerG, optimizerD, loss_func, epoch, device)
		test_loss = test(netG, netD, test_loader, loss_func, epoch, device)
		# scheduler.step()

		if not os.path.isdir('checkpoints/'):
			os.mkdir('checkpoints/')

		torch.save(netD.state_dict(), './checkpoints/'+str(name)+'_ModelD')
		torch.save(netG.state_dict(), './checkpoints/'+str(name)+'_ModelG')
		best_loss = test_loss

if __name__ == '__main__':
  main()




