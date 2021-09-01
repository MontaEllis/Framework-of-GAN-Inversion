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
    