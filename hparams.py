from transforms_config import car_transforms, normal_transforms

class hparams:
    description = 'PyTorch Training'
    output_dir = 'log/0831'

    img_size = 1024
    resize = True
    dataset_type = 'ffhq' # ['ffhq', 'celebahq', 'car', 'bggn']
    dataset_path = ''

    apply_init = True
    if apply_init:
        init_type = 'normal' # ['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none]

    backbone = 'GradualStyleEncoder' # ['GradualStyleEncoder', 'ResNetGradualStyleEncoder']
    epochs_per_checkpoint = 10
    latest_checkpoint_file = 'checkpoint_latest.pt'
    epochs = 50000
    batch = 5
    ckpt = None

    init_lr = 0.001

    # loss
    loss_lambda_mse = 1
    loss_lambda_lpips = 0.8
    loss_lambda_id = 0.1

    
    arc_model_path = ''
    moco_model_path = ''
    circular_face_model_paths = ''
    weight_path_pytorch = ''

    mtcnn_path_pnet = ''
    mtcnn_path_rnet = ''
    mtcnn_path_onet = ''

    if dataset_type == 'car':
        transform = car_transforms.get_transforms()
    else:
        transform = normal_transforms.get_transforms()

    optimizer_mode = 'adam' # ['adam', 'sgd', 'radam', 'lookahead', 'ranger']
    scheduler_mode = 'StepLR' # ['StepLR', 'MultiStepLR', 'ReduceLROnPlateau']

    open_warn_up = True
    if open_warn_up:
        warn_up_strategy = 'cos' # ['cos', 'linear', 'constant']
        num_warmup = 3

    # for save
    norm = True
    row = 1
    rangee = (-1,1)

    