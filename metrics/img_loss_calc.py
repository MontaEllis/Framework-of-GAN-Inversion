# modify from https://github.com/eladrich/pixel2style2pixel/blob/master/scripts/calc_losses_on_images.py
from argparse import ArgumentParser
import os
import json
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from criteria.lpips.lpips import LPIPS
from data_function import GTResDataset

from piqa import PSNR, TV, SSIM, MS_SSIM, GMSD, MS_GMSD, MDSI, HaarPSI, VSI, FSIM 

def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--mode', type=str, default='ms_ssim', choices=['lpips', 'l2', 'psnr', 'tv', 'ssim', 'ms_ssim', 'gmsd', 'ms_gmsd', 'mdsi', 'haarpsi', 'vsi', 'fsim'])
	parser.add_argument('--data_path', type=str, default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=4)
	args = parser.parse_args()
	return args


def run(args):



	if args.mode == 'lpips' or args.mode == 'l2':
		transform = transforms.Compose([transforms.Resize((256, 256)),
										transforms.ToTensor(),
										transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
	else:
		transform = transforms.Compose([transforms.Resize((256, 256)),
										transforms.ToTensor()])		

	print('Loading dataset')
	dataset = GTResDataset(root_path=args.data_path,
	                       gt_dir=args.gt_path,
						   transform=transform)

	dataloader = DataLoader(dataset,
	                        batch_size=args.batch_size,
	                        shuffle=False,
	                        num_workers=int(args.workers),
	                        drop_last=True)

	if args.mode == 'lpips':
		loss_func = LPIPS(net_type='alex')
	elif args.mode == 'l2':
		loss_func = torch.nn.MSELoss()
	elif args.mode == 'psnr':
		loss_func = PSNR()
	elif args.mode == 'tv':
		loss_func = TV()
	elif args.mode == 'ssim':
		loss_func = SSIM()
	elif args.mode == 'ms_ssim':
		loss_func = MS_SSIM()
	elif args.mode == 'gmsd':
		loss_func = GMSD()
	elif args.mode == 'ms_gmsd':
		loss_func = MS_GMSD()
	elif args.mode == 'mdsi':
		loss_func = MDSI()
	elif args.mode == 'haarpsi':
		loss_func = HaarPSI()
	elif args.mode == 'vsi':
		loss_func = VSI()
	elif args.mode == 'fsim':
		loss_func = FSIM()	
	else:
		raise Exception('Not a valid mode!')
	loss_func.cuda()

	global_i = 0
	scores_dict = {}
	all_scores = []
	for result_batch, gt_batch in tqdm(dataloader):
		# print(result_batch)
		# print(gt_batch)
		for i in range(args.batch_size):
			if args.mode == 'lpips' or args.mode == 'l2':
				loss = float(loss_func(result_batch[i:i+1].cuda(), gt_batch[i:i+1].cuda()))
			else:
				loss = loss_func(result_batch.cuda(),gt_batch.cuda())

			all_scores.append(loss)
			im_path = dataset.pairs[global_i][0]
			if args.mode == 'lpips' or args.mode == 'l2':
				scores_dict[os.path.basename(im_path)] = loss
			else:
				scores_dict[os.path.basename(im_path)] = loss.cpu()
			global_i += 1
			
			if args.mode == 'lpips' or args.mode == 'l2':
				continue
			else:
				break
	all_scores = list(scores_dict.values())
	 
	mean = np.mean(all_scores)
	std = np.std(all_scores)
	result_str = 'Average loss is {:.4f}+-{:.4f}'.format(mean, std)
	print('Finished with ', args.data_path)
	print(result_str)

	# out_path = os.path.join(os.path.dirname(args.data_path), 'inference_metrics')
	# if not os.path.exists(out_path):
	# 	os.makedirs(out_path)

	# with open(os.path.join(out_path, 'stat_{}.txt'.format(args.mode)), 'w') as f:
	# 	f.write(result_str)
	# with open(os.path.join(out_path, 'scores_{}.json'.format(args.mode)), 'w') as f:
	# 	json.dump(scores_dict, f)


if __name__ == '__main__':
	args = parse_args()
	run(args)