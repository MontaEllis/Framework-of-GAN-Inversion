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
from weights_init.weight_init_normal import weights_init_normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


np_str_obj_array_pattern = re.compile(r'[SaUO]')

face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch, help='batch-size')  

    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")


    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')

    return parser



def train():

    parser = argparse.ArgumentParser(description=hp.description)
    parser = parse_training_args(parser)
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


    if hp.apply_init:
        model.apply(weights_init_normal)


    model = torch.nn.DataParallel(model, device_ids=devicess)


    # params = list(model.parameters()) + list(class_generate.g_ema.parameters())
    params = list(model.parameters())
    if hp.optimizer_mode == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.init_lr, betas=(0.95, 0.999))
    elif hp.optimizer_mode == 'sgd':  
        optimizer = torch.optim.SGD(params, lr=args.init_lr, momentum=0.9, weight_decay=0.0005)
    elif hp.optimizer_mode == 'radam':
        optimizer = RAdam(params, lr=args.init_lr, betas=(0.95, 0.999))
    elif hp.optimizer_mode == 'lookahead':
        optimizer = Lookahead(params)
    elif hp.optimizer_mode == 'ranger':
        optimizer = Ranger(params, lr=args.init_lr)
    else:
        raise Exception('Optimizer error!')
    
    




    if hp.scheduler_mode == 'StepLR':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif hp.scheduler_mode == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=[3,6,9], gamma=0.1)
    elif hp.scheduler_mode == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, threshold=0.99, mode='min', patience=2, cooldown=5)
    else:
        raise Exception('Scheduler error!')

    if hp.open_warn_up:
        scheduler = WarmupLR(scheduler, init_lr=hp.init_lr, num_warmup=hp.num_warmup, warmup_strategy=hp.warn_up_strategy)



    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])

        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]

    else:
        elapsed_epochs = 0


    # model cuda
    model.cuda()

    from criteria import all_loss
    criterion = all_loss.Base_Loss()

    writer = SummaryWriter(args.output_dir)




    from data_function import ImageData

    train_dataset = ImageData(hp.dataset_path, hp.transform['transform_train'])
    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

    

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)


    model.train()

    for epoch in range(1, epochs + 1):
        

        print("epoch:"+str(epoch))
        epoch += elapsed_epochs

   
        for i, batch in enumerate(train_loader):
            

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            img = batch.cuda()
            if hp.resize:
                img = face_pool(img)
            outputs = model(img)
    
            predicts = class_generate.generate_from_synthesis(outputs,None,randomize_noise=True,return_latents=True)

            if hp.dataset_type == 'car':
                predicts = predicts[:, :, 32:224, :]

            # torch.set_grad_enabled(True)
            optimizer.zero_grad()


            loss_all,loss_mse,loss_lpips,loss_per = criterion(img,predicts)
            ## log
            writer.add_scalar('Refine/Loss', loss_all.item(), iteration)
            writer.add_scalar('Refine/loss_mse', loss_mse.item(), iteration)
            writer.add_scalar('Refine/loss_lpips', loss_lpips.item(), iteration)
            writer.add_scalar('Refine/loss_per', loss_per.item(), iteration)
            loss_all.backward()
            optimizer.step()
            print("loss:"+str(loss_all.item()))
            print('lr:'+str(scheduler._last_lr[0]))
            iteration += 1
        scheduler.step()

        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:
            torch.save(
                {
                    
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
        
            with torch.no_grad():
                utils.save_image(
                    predicts,
                    os.path.join(args.output_dir,("step-{}-predict.png").format(epoch)),
                    nrow=hp.row,
                    normalize=hp.norm,
                    range=hp.rangee,
                )
                
    writer.close()



   

if __name__ == '__main__':
    train()
