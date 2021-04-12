# Enhanced CycleGAN Dehazing Network
We focus on unpaired single image dehazing and reduce the image dehazing problem to an unpaired image-to-image translation and propose an **E**nhanced **C**ycleGAN **D**ehazing **N**etwork (ECDN). We enhance CycleGAN from different angles for the dehazing purpose. We employ a global-local discriminator structure to deal with spatially varying haze. We define self-regularized color loss and utilize it along with perceptual loss to generate more realistic and visually pleasing images. We use an encoder-decoder architecture with residual blocks in the generator with skip connections so that the network better preserves the details. Through an ablation study, we demonstrate the effectiveness of different modules in the performance of the proposed network. Our extensive experiments over two benchmark datasets show that our network outperforms previous work in terms of PSNR and SSIM. Paper is [here](http://vlm1.uta.edu/~athitsos/publications/anvari_visapp2021.pdf).

# ECDN Architecture

## Overview
![alt text](https://github.com/zanvari/ecdn/blob/main/figs/ecdn1.png?raw=true)

## Generator Architecture

![alt text](https://github.com/zanvari/ecdn/blob/main/figs/ecdn2.png?raw=true)

# Results

![alt text](https://github.com/zanvari/ecdn/blob/main/figs/results-ecdn.png?raw=true)

# Publication
If you find this work useful for you, please cite:

    @InProceedings{anvari_visapp2021,
        author    = {Anvari, Zahra and Athitsos, Vassilis},
        title     = {Enhanced CycleGAN Dehazing Network},
        booktitle = {Proceedings of the 16th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)},
        month     = {February},
        year      = {2021},
        volume    = {4},
        pages     = {193-202}
    }
