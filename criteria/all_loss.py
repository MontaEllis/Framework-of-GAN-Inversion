from torch import nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from PIL import Image
from torchvision import utils
from torch.autograd import Variable
import torch.autograd as autograd
from hparams import hparams as hp
from criteria import id_loss, moco_loss

class Base_Loss(nn.Module):
    def __init__(self):
        super(Base_Loss,self).__init__()
        import lpips
        # self.loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

        self.id_loss = id_loss.IDLoss().cuda().eval()
        self.moco_loss = moco_loss.MocoLoss()


        self.criterion_mse = nn.MSELoss()


    def forward(self, gt,predict_images):
            
        loss_mse = self.criterion_mse(gt, predict_images)
        loss_lpips = self.loss_fn_vgg(gt,predict_images)[0][0][0][0]

        if hp.dataset_type == 'car':
            loss_per = self.moco_loss(predict_images,gt,gt)[0]
        else:
            loss_per = self.id_loss(predict_images,gt,gt)[0]

        loss_all = loss_mse + loss_lpips + loss_per

        return loss_all,loss_mse,loss_lpips,loss_per
