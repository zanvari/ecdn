import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import random
import util.util as util
from skimage.filters import gaussian
import numpy as np
import torch.nn.functional as f
import math

def color_loss(image1, image2, len_reg=0):
    
    vec1 = torch.reshape(image1, (-1, 3))
    vec2 = torch.reshape(image2, (-1, 3))
    clip_value = 0.999999
    norm_vec1 = f.normalize(vec1, 1)
    norm_vec2 = f.normalize(vec2, 1)
    dot = torch.sum(norm_vec1*norm_vec2, 1)
    dot = torch.clamp(dot, -clip_value, clip_value)
    angle = torch.acos(dot) * (180/math.pi)

    return torch.mean(angle)

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.patchD:
            self.loss_names = ['D_A','D_P_A', 'G_A', 'cycle_A', 'idt_A', 'D_B','D_P_B', 'G_B', 'cycle_B', 'idt_B']

        else:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        if self.opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)

            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()

            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()

            for param in self.vgg.parameters():
                param.requires_grad = False

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            if self.opt.patchD:
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_P_A','D_P_B']
            else:
                self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            if self.opt.patchD:
                #use_sigmoid = opt.no_lsgan
                self.netD_P_A = networks.define_D( opt.input_nc, opt.ndf, opt.netD, opt.n_layers_patchD, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_P_B = networks.define_D( opt.input_nc, opt.ndf, opt.netD, opt.n_layers_patchD, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            #self.optimizer_G = torch.optim.RMSprop(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, alpha=0.99, weight_decay=0, eps=1e-08, momentum=0, centered=False)
            #self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, alpha=0.99, weight_decay=0, eps=1e-08, momentum=0, centered=False)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            if self.opt.patchD:
                #self.optimizer_D_P = torch.optim.Adam(self.netD_P_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_P = torch.optim.Adam(itertools.chain(self.netD_P_A.parameters(), self.netD_P_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.patchD:
                self.optimizers.append(self.optimizer_D_P) #for both D_P_A, D_P_B

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        #self.real_A = Variable(self.input_A)
        #self.real_B = Variable(self.input_B)

        if self.opt.patchD:
            w_A = self.real_A.size(3)
            h_A = self.real_A.size(2)
            
            w_B = self.real_B.size(3)
            h_B = self.real_B.size(2)

            w_offset_A = random.randint(0, max(0, w_A - self.opt.patchSize - 1))
            h_offset_A = random.randint(0, max(0, h_A - self.opt.patchSize - 1))

            w_offset_B = random.randint(0, max(0, w_B - self.opt.patchSize - 1))
            h_offset_B = random.randint(0, max(0, h_B - self.opt.patchSize - 1))

            self.rec_patch_A = self.rec_A[:,:, h_offset_A:h_offset_A + self.opt.patchSize, w_offset_A:w_offset_A + self.opt.patchSize]
            self.fake_patch_B = self.fake_B[:,:, h_offset_B:h_offset_B + self.opt.patchSize, w_offset_B:w_offset_B + self.opt.patchSize]
            self.real_patch_B = self.real_B[:,:, h_offset_B:h_offset_B + self.opt.patchSize, w_offset_B:w_offset_B + self.opt.patchSize]
            self.input_patch_A = self.real_A[:,:, h_offset_A:h_offset_A + self.opt.patchSize, w_offset_A:w_offset_A + self.opt.patchSize]

            self.rec_patch_B = self.rec_B[:,:, h_offset_B:h_offset_B + self.opt.patchSize, w_offset_B:w_offset_B + self.opt.patchSize]
            self.fake_patch_A = self.fake_A[:,:, h_offset_A:h_offset_A + self.opt.patchSize, w_offset_A:w_offset_A + self.opt.patchSize]
            self.real_patch_A = self.real_A[:,:, h_offset_A:h_offset_A + self.opt.patchSize, w_offset_A:w_offset_A + self.opt.patchSize]
            self.input_patch_B = self.real_B[:,:, h_offset_B:h_offset_B + self.opt.patchSize, w_offset_B:w_offset_B + self.opt.patchSize]


        if self.opt.patchD_3 > 0:
            self.rec_patch_1 = []
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []

            self.rec_patch_2 = []
            self.fake_patch_2 = []
            self.real_patch_2 = []
            self.input_patch_2 = []

            w_1 = self.real_A.size(3)
            h_1 = self.real_A.size(2)
            #print('w_1, h_1: ', w_1, h_1)


            w_2 = self.real_B.size(3)
            h_2 = self.real_B.size(2)
            #print('w_2, h_2: ', w_2, h_2)

            for i in range(self.opt.patchD_3):

                w_offset_1 = random.randint(0, max(0, w_1 - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h_1 - self.opt.patchSize - 1))

                w_offset_2 = random.randint(0, max(0, w_2 - self.opt.patchSize - 1))
                h_offset_2 = random.randint(0, max(0, h_2 - self.opt.patchSize - 1))

                self.rec_patch_1.append(self.rec_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize, w_offset_1:w_offset_1 + self.opt.patchSize])
                self.fake_patch_1.append(self.fake_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize, w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize, w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:,:, h_offset_1:h_offset_1 + self.opt.patchSize, w_offset_1:w_offset_1 + self.opt.patchSize])

                self.rec_patch_2.append(self.rec_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize, w_offset_2:w_offset_2 + self.opt.patchSize])
                self.fake_patch_2.append(self.fake_A[:,:, h_offset_2:h_offset_2 + self.opt.patchSize, w_offset_2:w_offset_2 + self.opt.patchSize])
                self.real_patch_2.append(self.real_A[:,:, h_offset_2:h_offset_2 + self.opt.patchSize, w_offset_2:w_offset_2 + self.opt.patchSize])
                self.input_patch_2.append(self.real_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize, w_offset_2:w_offset_2 + self.opt.patchSize])


    #def backward_D_basic(self, netD, real, fake, use_ragan):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        ## Real
        #pred_real = netD(real)
        #loss_D_real = self.criterionGAN(pred_real, True)
        ## Fake
        #pred_fake = netD(fake.detach())
        #loss_D_fake = self.criterionGAN(pred_fake, False)
        ## Combined loss and calculate gradients
        #loss_D = (loss_D_real + loss_D_fake) * 0.5
        ##loss_D.backward()
        #return loss_D

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, real.data, fake.data)

        elif self.opt.use_ragan and use_ragan:
            #print('ragan.....')
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) + self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, True)

    def backward_D_P_A(self):

        if self.opt.hybrid_loss:
            loss_D_P_A = self.backward_D_basic(self.netD_P_A, self.real_patch_B, self.fake_patch_B, False)

            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P_A += self.backward_D_basic(self.netD_P_A, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P_A = loss_D_P_A/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P_A = loss_D_P_A
        else:
            loss_D_P_A = self.backward_D_basic(self.netD_P_A, self.real_patch_B, self.fake_patch_B, True)

            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P_A += self.backward_D_basic(self.netD_P_A, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P_A = loss_D_P_A/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P_A = loss_D_P_A
        if self.opt.D_P_times2:
            self.loss_D_P_A = self.loss_D_P_A*2
        #self.loss_D_P_A.backward()
        

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, True)

    def backward_D_P_B(self):

        if self.opt.hybrid_loss:
            loss_D_P_B = self.backward_D_basic(self.netD_P_B, self.real_patch_A, self.fake_patch_A, False)

            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P_B += self.backward_D_basic(self.netD_P_B, self.real_patch_2[i], self.fake_patch_2[i], False)
                self.loss_D_P_B = loss_D_P_B/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P_B = loss_D_P_B
        else:
            loss_D_P_B = self.backward_D_basic(self.netD_P_B, self.real_patch_A, self.fake_patch_A, True)

            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P_B += self.backward_D_basic(self.netD_P_B, self.real_patch_2[i], self.fake_patch_2[i], True)
                self.loss_D_P_B = loss_D_P_B/float(self.opt.patchD_3 + 1)

            else:
                self.loss_D_P_B = loss_D_P_B

        if self.opt.D_P_times2:
            self.loss_D_P_B = self.loss_D_P_B*2
        #self.loss_D_P_B.backward()

    
    def backward_G(self):
        if self.opt.patchD:
            pred_fake_A = self.netD_P_A(self.fake_A)
            pred_fake_B = self.netD_P_B(self.fake_B)
        
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
             # G_A should be identity if real_B is fed: ||G_A(B) - B||
             self.idt_A = self.netG_A(self.real_B)
             self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
             # G_B should be identity if real_A is fed: ||G_B(A) - A||
             self.idt_B = self.netG_B(self.real_A)
             self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.opt.use_ragan:
            #print('ragan.....')

            pred_real_A = self.netD_A(self.real_A)
            pred_fake_A = self.netD_A(self.fake_A)

            pred_real_B = self.netD_B(self.real_B)
            pred_fake_B = self.netD_B(self.fake_B)

            # GAN loss D_A(G_A(A))
            self.loss_G_A = (self.criterionGAN(pred_real_A - torch.mean(pred_fake_A), True) + self.criterionGAN(pred_fake_A - torch.mean(pred_real_A), False)) / 2
            # GAN loss D_B(G_B(B))
            self.loss_G_B = (self.criterionGAN(pred_real_B - torch.mean(pred_fake_B), True) + self.criterionGAN(pred_fake_B - torch.mean(pred_real_B), False)) / 2

        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #print(lambda_A, lambda_B)
        #######################patchD loss############################################

        loss_G_A = 0
        loss_G_B = 0

        if self.opt.patchD:
            pred_fake_patch_A = self.netD_P_A.forward(self.fake_patch_A)
            pred_fake_patch_B = self.netD_P_B.forward(self.fake_patch_B)

            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch_B, True)
                loss_G_B += self.criterionGAN(pred_fake_patch_A, True)

            else:
                pred_real_patch_A = self.netD_P_A.forward(self.real_patch_A)
                pred_real_patch_B = self.netD_P_B.forward(self.real_patch_B)
                loss_G_A += (self.criterionGAN(pred_real_patch_B - torch.mean(pred_fake_patch_B), False) + self.criterionGAN(pred_fake_patch_B - torch.mean(pred_real_patch_B), True)) / 2
                loss_G_B += (self.criterionGAN(pred_real_patch_A - torch.mean(pred_fake_patch_A), False) + self.criterionGAN(pred_fake_patch_A - torch.mean(pred_real_patch_A), True)) / 2

        ############################################################################################
        if self.opt.patchD and self.opt.patchD_3 > 0:
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P_A.forward(self.fake_patch_1[i])
                pred_fake_patch_2 = self.netD_P_B.forward(self.fake_patch_2[i])

                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_2, True)
                    loss_G_B += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    #print(len(self.real_patch_1[i]))
                    pred_real_patch_1 = self.netD_P_A(self.real_patch_1[i])
                    pred_real_patch_2 = self.netD_P_B(self.real_patch_2[i])
                    loss_G_A += (self.criterionGAN(pred_real_patch_2 - torch.mean(pred_fake_patch_2), False) + self.criterionGAN(pred_fake_patch_2 - torch.mean(pred_real_patch_2), True)) / 2
                    loss_G_B += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) + self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2

            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)
                self.loss_G_B += loss_G_B/float(self.opt.patchD_3 + 1)

            else:
                self.loss_G_A += loss_G_A/float(self.opt.patchD_3 + 1)*2
                self.loss_G_B += loss_G_B/float(self.opt.patchD_3 + 1)*2

        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
                self.loss_G_B += loss_G_B
            else:
                self.loss_G_A += loss_G_A*2
                self.loss_G_B += loss_G_B*2


        if self.opt.vgg > 0:
            #self.loss_vgg_b_A = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            #self.loss_vgg_b_B = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_A, self.real_B) * self.opt.vgg if self.opt.vgg > 0 else 0
           
            self.loss_vgg_b_A = self.vgg_loss.compute_vgg_loss(self.vgg, self.rec_A, self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            self.loss_vgg_b_B = self.vgg_loss.compute_vgg_loss(self.vgg, self.rec_B, self.real_B) * self.opt.vgg if self.opt.vgg > 0 else 0

            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    #loss_vgg_patch_A = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_patch_B, self.input_patch_A) * self.opt.vgg
                    #loss_vgg_patch_B = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_patch_A, self.input_patch_B) * self.opt.vgg

                    loss_vgg_patch_A = self.vgg_loss.compute_vgg_loss(self.vgg, self.rec_patch_A, self.input_patch_A) * self.opt.vgg
                    loss_vgg_patch_B = self.vgg_loss.compute_vgg_loss(self.vgg, self.rec_patch_B, self.input_patch_B) * self.opt.vgg

                else:
                    #loss_vgg_patch_A = self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.fake_patch_B, self.input_patch_A) * self.opt.vgg
                    #loss_vgg_patch_B = self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.fake_patch_A, self.input_patch_B) * self.opt.vgg

                    loss_vgg_patch_A = self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.rec_patch_A, self.input_patch_A) * self.opt.vgg
                    loss_vgg_patch_B = self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.rec_patch_B, self.input_patch_B) * self.opt.vgg


            if self.opt.patch_vgg and self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    if not self.opt.IN_vgg:
                        #loss_vgg_patch_A += self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        #loss_vgg_patch_B += self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_patch_2[i], self.input_patch_2[i]) * self.opt.vgg

                        loss_vgg_patch_A += self.vgg_loss.compute_vgg_loss(self.vgg, self.rec_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        loss_vgg_patch_B += self.vgg_loss.compute_vgg_loss(self.vgg, self.rec_patch_2[i], self.input_patch_2[i]) * self.opt.vgg

                    else:
                        #loss_vgg_patch_A += self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.fake_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        #loss_vgg_patch_B += self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.fake_patch_2[i], self.input_patch_2[i]) * self.opt.vgg

                        loss_vgg_patch_A += self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.rec_patch_1[i], self.input_patch_1[i]) * self.opt.vgg
                        loss_vgg_patch_B += self.vgg_patch_loss.compute_vgg_loss(self.vgg, self.rec_patch_2[i], self.input_patch_2[i]) * self.opt.vgg

                self.loss_vgg_b_A += loss_vgg_patch_A/float(self.opt.patchD_3 + 1)
                self.loss_vgg_b_B += loss_vgg_patch_B/float(self.opt.patchD_3 + 1)
                
            elif self.opt.patchD:
                self.loss_vgg_b_A += loss_vgg_patch_A
                self.loss_vgg_b_B += loss_vgg_patch_B

            #self.loss_G = self.loss_G_A + self.loss_vgg_b

           # realA = util.tensor2img_batch(self.real_A) 
            #recA = util.tensor2img_batch(self.rec_A)
            
            realB = util.tensor2img_batch(self.real_B) 
            recB = util.tensor2img_batch(self.rec_B)

            #print("realB: ", realB)
            #print("fakeB: ", fakeB)
#(:,:,1)
            color_loss_A = 0
            color_loss_B = 0

            #color_loss_A = color_loss(self.real_A, self.rec_A)
            color_loss_B = color_loss(self.real_B, self.rec_B)
            
            #print("color_A: ", (color_loss_A).data[0]*0.05, "color_B: ", (color_loss_B).data[0]*0.05) 
            #print( "G_A: ", (self.loss_G_A).data[0], "G_B: ", (self.loss_G_B).data[0], "cycle_A: ", (self.loss_cycle_A).data[0], "cycle_B: ", (self.loss_cycle_B).data[0], "vgg_A: ", (self.loss_vgg_b_A).data[0], "vgg_B: ", (self.loss_vgg_b_B).data[0])

            #for r1,f1 in zip(realA, recA):
                #real_A_blurred_r = gaussian(r1[:, :, 0], sigma=1, multichannel=True)
                #rec_A_blurred_r = gaussian(f1[:, :, 0], sigma=1, multichannel=True)
                #color_loss_A += np.mean((rec_A_blurred_r - real_A_blurred_r) ** 2)

                #real_A_blurred_g = gaussian(r1[:, :, 1], sigma=1, multichannel=True)
                #rec_A_blurred_g = gaussian(f1[:, :, 1], sigma=1, multichannel=True)
                #color_loss_A += np.mean((rec_A_blurred_g - real_A_blurred_g) ** 2)

                #real_A_blurred_b = gaussian(r1[:, :, 2], sigma=1, multichannel=True)
                #rec_A_blurred_b = gaussian(f1[:, :, 2], sigma=1, multichannel=True)
                #color_loss_A += np.mean((rec_A_blurred_b - real_A_blurred_b) ** 2)


            #for r2,f2 in zip(realB, recB):
                #real_B_blurred_r = gaussian(r2[:, :, 0], sigma=1, multichannel=True)
                #rec_B_blurred_r = gaussian(f2[:, :, 0], sigma=1, multichannel=True)
                #color_loss_B += np.mean((rec_B_blurred_r - real_B_blurred_r) ** 2)

                #real_B_blurred_g = gaussian(r2[:, :, 1], sigma=1, multichannel=True)
                #rec_B_blurred_g = gaussian(f2[:, :, 1], sigma=1, multichannel=True)
                #color_loss_B += np.mean((rec_B_blurred_g - real_B_blurred_g) ** 2)

                #real_B_blurred_b = gaussian(r2[:, :, 2], sigma=1, multichannel=True)
                #rec_B_blurred_b = gaussian(f2[:, :, 2], sigma=1, multichannel=True)
                #color_loss_B += np.mean((rec_B_blurred_b - real_B_blurred_b) ** 2)



                #print("color loss: ",color_loss)
                #print(self.loss_G_A, self.loss_vgg_b*vgg_w)

            #if epoch % 5 == 0:
            #print("color loss A: ",0.1*color_loss_A)
            #print("color loss B: ",0.1*color_loss_B)
                #print("loss G_A: ",self.loss_G_A)
                #print("vgg_loss: ",self.loss_vgg_b*vgg_w)
            #self.loss_G = self.loss_G_A + self.loss_vgg_b*vgg_w + 0.3*color_loss
                #print("loss_G: ", self.loss_G)


        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_vgg_b_A + self.loss_vgg_b_B + 0.05*color_loss_B
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B 
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        if self.opt.patchD:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_P_A, self.netD_P_B], False)  # Ds require no gradients when optimizing Gs
        else:
            self.set_requires_grad([self.netD_A, self.netD_B], False)

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        #####################################################
        if self.opt.patchD:
            # self.forward()
            self.set_requires_grad([self.netD_P_A, self.netD_P_B], True)
            self.optimizer_D_P.zero_grad()
            self.backward_D_P_A()
            self.backward_D_P_B()
            #self.optimizer_D.step()
            self.optimizer_D_P.step() #update D_P_A, D_P_B
        #####################################################

        #self.optimizer_D.step()  # update D_A and D_B's weights




