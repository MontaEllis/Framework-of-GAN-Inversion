from glob import glob
from os.path import dirname, join, basename, isfile
import sys
import csv
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import os
from pathlib import Path
import argparse
import cv2
import sys
from sklearn.metrics import mean_squared_error, r2_score
import json
from torchvision import utils
from hparams import hparams as hp
from tqdm import tqdm

 

class ImageData(torch.utils.data.Dataset):
    def __init__(self, root_dir, transfom):


        self.transformss = transfom
        self.root_dir = root_dir
        self.root_dir_list = os.listdir(self.root_dir)



    def __len__(self):

        return len(self.root_dir_list)
    
    def __getitem__(self, index): 



        img_path = os.path.join(self.root_dir, self.root_dir_list[index])

        img = Image.open(img_path).convert('RGB')

        img = self.transformss(img)
        return img


class GTResDataset(torch.utils.data.Dataset):

	def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None):
		self.pairs = []
		for f in os.listdir(root_path):
			image_path = os.path.join(root_path, f)
			gt_path = os.path.join(gt_dir, f)
			if f.endswith(".jpg") or f.endswith(".png"):
				self.pairs.append([image_path, gt_path.replace('.png', '.jpg'), None])
		self.transform = transform
		self.transform_train = transform_train

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		from_path, to_path, _ = self.pairs[index]
		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')

		if self.transform:
			to_im = self.transform(to_im)
			from_im = self.transform(from_im)

		return from_im, to_im

        