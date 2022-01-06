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
import torch_fidelity
sys.path.append(".")
sys.path.append("..")

from data_function import GTResDataset
from cleanfid import fid

def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--data_path', type=str, default='results')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	parser.add_argument('--batch_size', type=int, default=4)
	args = parser.parse_args()
	return args


def run(args):


	metrics_dict = torch_fidelity.calculate_metrics(
		input1=args.data_path, 
		input2=args.gt_path, 
		cuda=True, 
		isc=True, 
		fid=True, 
		kid=True, 
		ppl=True,
		verbose=False,
	)

	
	print('Finished')
	print(metrics_dict)

	score_fid_clean = fid.compute_fid(args.gt_path, args.data_path,mode="clean")
	score_fid_tf = fid.compute_fid(args.gt_path, args.data_path,mode="legacy_tensorflow")
	score_fid_torch = fid.compute_fid(args.gt_path, args.data_path,mode="legacy_pytorch")
	score_kid_clean = fid.compute_kid(args.gt_path, args.data_path,mode="clean")
	score_kid_tf = fid.compute_kid(args.gt_path, args.data_path,mode="legacy_tensorflow")
	score_kid_torch = fid.compute_kid(args.gt_path, args.data_path,mode="legacy_pytorch")
	print('fid clean:',score_fid_clean)
	print('fid tf:',score_fid_tf)
	print('fid torch:',score_fid_torch)
	print('kid clean:',score_kid_clean)
	print('kid tf:',score_kid_tf)
	print('kid torch:',score_kid_torch)

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