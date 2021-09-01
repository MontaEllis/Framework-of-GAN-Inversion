# Framework of GAN Inversion

## Introcuction
* You can implement your own inversion idea using our repo. We offer a full range of tuning settings (in hparams.py), some excellent backbones and classics loss functions. You can modify the arch of network or loss easily.

## Recent Updates
* 2021.9.1 The simplfied framework of GAN Inversion is released.

## Requirements
* pip install git+git://github.com/lehduong/torch-warmup-lr.git
* PyTorch1.7

## Done
### Tuning Setting
* Apply_init
* Optimizer_mode
* Scheduler_mode
* Open_warn_up

### Backbone
* GradualStyleEncoder from [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
* ResNetGradualStyleEncoder from [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)


### Loss
* MSE
* LPIPS
* ID loss from [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
* Moco loss from [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

## TODO
- [ ] DDP
- [ ] More Backbones
- [ ] Metrics

## Acknowledgements
This repository is an unoffical PyTorch Framework of GAN Inversion and highly based on [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder), [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), [AliProducts](https://github.com/misads/AliProducts) and [stylegan2](https://github.com/NVlabs/stylegan2).Thank you for the above repo. Thank you to [Daiheng Gao](https://github.com/tomguluson92) and [Jie Zhang](https://scholar.google.com.hk/citations?user=gBkYZeMAAAAJ) for all the help I received.
