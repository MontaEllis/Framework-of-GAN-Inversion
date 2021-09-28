import os
from weights_init.weight_init_normal import weights_init_normal
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
devicess = [0,1,2]
import re
import time
import argparse
import numpy as np
from torch._six import container_abcs, string_classes, int_classes
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.distributed as dist
import math
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
from torchvision import utils
from hparams import hparams as hp
from torch.autograd import Variable
from torch_warmup_lr import WarmupLR
from optimizer.LookAhead import Lookahead
from optimizer.RAdam import RAdam
from optimizer.Ranger import Ranger
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


np_str_obj_array_pattern = re.compile(r'[SaUO]')

face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

def parse_testing_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save results')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Use the latest checkpoint in each epoch')

    testing = parser.add_argument_group('testing setup')
    testing.add_argument('--batch', type=int, default=1, help='batch-size')  

    testing.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    testing.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')

    return parser



def test():

    parser = argparse.ArgumentParser(description=hp.description)
    parser = parse_testing_args(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


    os.makedirs(args.output_dir, exist_ok=True)


    from stylegan2.stylegan2_infer import infer_face
    class_generate = infer_face(hp.weight_path_pytorch)


    n_styles = 2*int(math.log(hp.img_size, 2))-2
    if hp.backbone == 'GradualStyleEncoder':
        from models.fpn_encoders import GradualStyleEncoder
        model = GradualStyleEncoder(num_layers=50,n_styles=n_styles)
    elif hp.backbone == 'ResNetGradualStyleEncoder':
        from models.fpn_encoders import ResNetGradualStyleEncoder
        model = ResNetGradualStyleEncoder(n_styles=n_styles)
    else:
        Exception('Backbone error!')



    model = torch.nn.DataParallel(model, device_ids=devicess)



    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    # model cuda
    model.cuda()


    from data_function import ImageData

    test_dataset = ImageData(hp.dataset_path, hp.transform['transform_test'])
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

    



    model.eval()


   
    for i, batch in enumerate(test_loader):
        
        img = batch.cuda()
        if hp.resize:
            img = face_pool(img)
        outputs = model(img)

        predicts = class_generate.generate_from_synthesis(outputs,None,randomize_noise=False,return_latents=True)

        if hp.dataset_type == 'car':
            predicts = predicts[:, :, 32:224, :]

       
    
        with torch.no_grad():
            utils.save_image(
                predicts,
                os.path.join(args.output_dir,("step-{}-predict.png").format(i)),
                nrow=hp.row,
                normalize=hp.norm,
                range=hp.rangee,
            )
                

        with torch.no_grad():
            utils.save_image(
                img,
                os.path.join(args.output_dir,("step-{}-origin.png").format(i)),
                nrow=hp.row,
                normalize=hp.norm,
                range=hp.rangee,
            )

   

if __name__ == '__main__':
    test()
