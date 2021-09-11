import torch
import torch.nn as nn
import utils.util as util
from models.dips import ImageDIP
from models.backbones.edsr import EDSR
from models.kernel_encoding.kernel_wizard import KernelExtractor
from tqdm import tqdm
import cv2
from models.losses.hyper_laplacian_penalty import HyperLaplacianPenalty
from models.losses.perceptual_loss import PerceptualLoss
from models.losses.ssim_loss import SSIM
from torch.optim.lr_scheduler import StepLR
from data.common import downsample, conv


class BlindSR:
    def __init__(self, opt):
        self.opt = opt
        self.ssim_loss = SSIM().cuda()
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.perceptual_loss = PerceptualLoss().cuda()
        self.laplace_penalty = HyperLaplacianPenalty(3, 0.66).cuda()
        self.dip = ImageDIP(opt["network"]["DIP"]).cuda()
        self.SR = EDSR(opt["network"]["SR"]).cuda()
        self.netG = KernelExtractor(opt["network"]["KernelExtractor"]).cuda()
        self.load()

    def prepare_DIP(self, size):
        self.random_x = util.get_noise(8, "noise", size).cuda()

    def reset_optimizers(self):
        self.x_optimizer = torch.optim.Adam(self.dip.parameters(), lr=self.opt["x_lr"])
        self.x_scheduler = StepLR(self.x_optimizer, step_size=self.opt["num_iters"] // 5, gamma=0.7)
        # self.k_optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
        # self.k_scheduler = StepLR(self.k_optimizer, step_size=self.opt["num_iters"] // 5, gamma=0.7)

    def warmup(self, warmup_x):
        # Input vector of DIPs is sampled from N(z, I)
        reg_noise_std = self.opt["reg_noise_std"]

        print("Warming up DIP")

        for step in tqdm(range(self.opt["num_warmup_iters"])):
            self.x_optimizer.zero_grad()
            dip_zx_rand = self.random_x + reg_noise_std * torch.randn_like(self.random_x).cuda()
            x = self.dip(dip_zx_rand)

            loss = self.mse(x, warmup_x)
            loss.backward()
            self.x_optimizer.step()

    def lr_from_hr(self, hr, k):
        hr_blur = conv(hr.permute(1,0,2,3), k, padding=7)
        lr = downsample(hr_blur)
        return lr


    def SR_step(self, lr, hr):
        """Enhance resolution
        Args:
            lr: Low-resolution image
        """
        # lr = util.img2tensor(lr).unsqueeze(0).cuda()
        # hr = util.img2
        print(lr.shape, hr.shape)
        size = [hr.shape[2], hr.shape[3]]

        print("Step Super-resolution")
        with torch.no_grad():
            hr_blur = self.SR(lr)

        self.prepare_DIP(size)
        self.reset_optimizers()

        self.warmup(hr_blur)
     
        # Input vector of DIPs is sampled from N(z, I)

        print("Deblurring")
        reg_noise_std = self.opt["reg_noise_std"]
        for step in tqdm(range(self.opt["num_iters"])):
            # dip_zx_rand = self.random_x + reg_noise_std * torch.randn_like(self.random_x).cuda()

            self.x_optimizer.zero_grad()
            self.x_scheduler.step()

            hr_pred = self.dip(self.random_x)
            # with torch.no_grad():
            k_pred, blur_pred = self.netG(hr_pred, hr_blur)
            
            # print(k_pred.shape)
            # print(blur_pred.shape)

            lr_pred = downsample(blur_pred)

            if step < self.opt["num_iters"] // 2:
                total_loss = 6e-1 * self.l1(lr_pred, lr)
                total_loss += 1 - self.ssim_loss(lr_pred, lr)
                total_loss += 5e-5 * torch.norm(k_pred)
                total_loss += 2e-2 * self.laplace_penalty(hr_pred)
            else:
                total_loss = self.l1(lr_pred, lr)
                total_loss += 5e-2 * self.laplace_penalty(hr_pred)
                total_loss += 5e-4 * torch.norm(k_pred)

            total_loss.backward()

            self.x_optimizer.step()
            # self.k_optimizer.step()

            # debugging
            # if step % 100 == 0:
            #     print(torch.norm(k))
            #     print(f"{self.k_optimizer.param_groups[0]['lr']:.3e}")
        img_blur = util.tensor2img(blur_pred.detach())
        cv2.imwrite('./hr_blur.png', img_blur)
        return util.tensor2img(hr_pred.detach())

    def load(self):
        """
        load or initialize network
        """
        SR_path = self.opt['solver']['pretrainedSR_path']
        netG_path = self.opt['solver']['pretrainednetG_path']
        if SR_path is None: raise ValueError("[Error] The 'pretrainedSR_path' does not declarate in *.json")
        if netG_path is None: raise ValueError("[Error] The 'pretrainednetG_path' does not declarate in *.json")

        print('===> Loading SR module from [%s]...' % SR_path)
    
        checkpoint = torch.load(SR_path)
        if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
        load_func = self.SR.load_state_dict
        load_func(checkpoint)

        print('===> Loading netG module from [%s]...' % netG_path)
    
        checkpoint = torch.load(netG_path)
        if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
        load_func = self.netG.load_state_dict
        load_func(checkpoint)

