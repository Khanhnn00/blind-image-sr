import torch
import torch.nn as nn
import utils.util as util
from models.dips import ImageDIP
from models.backbones.edsr import EDSR
from models.kernel_encoding.kernel_wizard import KernelExtractor
from models.sr.IDK import IDK
from tqdm import tqdm
import cv2
from models.losses.hyper_laplacian_penalty import HyperLaplacianPenalty
from models.losses.perceptual_loss import PerceptualLoss
from models.losses.ssim_loss import SSIM
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from data.common import downsample, conv
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F


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
        self.scale=self.opt["scale"]

    def prepare_DIP(self, size):
        self.random_x = util.get_noise(8, "noise", size).cuda()
    
    def reset_optimizers(self):
        self.x_optimizer = torch.optim.Adam(self.dip.parameters(), lr=self.opt["x_lr"])
        # self.x_scheduler = MultiStepLR(self.x_optimizer, [1, (self.opt["num_iters"] // 5)], gamma=[0.995, 0.7])
        self.x_scheduler = StepLR(self.x_optimizer, step_size=self.opt["num_iters"] // 2, gamma=0.7)
        # self.k_optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0001)
        # self.k_scheduler = StepLR(self.k_optimizer, step_size=self.opt["num_iters"] // 5, gamma=0.7)

    def warmup(self, warmup_x):
        # Input vector of DIPs is sampled from N(z, I)
        # reg_noise_std = self.opt["reg_noise_std"]

        print("Warming up DIP")

        for step in tqdm(range(self.opt["num_warmup_iters"])):
            self.x_optimizer.zero_grad()
            # dip_zx_rand = self.random_x + reg_noise_std * torch.randn_like(self.random_x).cuda()
            x = self.dip(self.random_x)

            loss = self.mse(x, warmup_x)
            print(loss)
            if loss.item() < 10:
                return
            loss.backward()
            self.x_optimizer.step()
        res = util.tensor2img(x.detach())
        cv2.imwrite('./after_warmup.png', res)

    def extract(self, hr, hr_blur):
        k, blur = self.netG(hr.cuda(), hr_blur.cuda())
        save_image(k, './test_k.png',nrow=1,  normalize=True)
        return util.tensor2img(blur.detach())

    def SR_step(self, lr, k):
        """Enhance resolution
        Args:
            lr: Low-resolution image
        """
        # lr = util.img2tensor(lr).unsqueeze(0).cuda()
        # hr = util.img2
        self.SR.eval()
        print(lr.shape)
        size = [lr.shape[2]*self.opt["scale"], lr.shape[3]*self.opt["scale"]]

        print("Step Super-resolution")
        with torch.no_grad():
            hr_blur = self._overlap_crop_forward(lr, n_GPUs=1)
        # print(hr_blur.mean(), hr_blur.max(), hr_blur.min())
        img_blur = hr_blur.data[0].float().cpu()
        img_blur = util.Tensor2np([img_blur], 255)[0]
        cv2.imwrite('./hr_blur.png', cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))

        self.prepare_DIP(size)
        self.reset_optimizers()

        self.warmup(hr_blur)
     
        # Input vector of DIPs is sampled from N(z, I)

        print("Deblurring")
        reg_noise_std = self.opt["reg_noise_std"]
        self.x_optimizer.param_groups[0]['lr'] = 5e-4
        for step in tqdm(range(self.opt["num_iters"])):
            # print('Current LR: {}'.format(self.x_optimizer.param_groups[0]['lr']))
            # dip_zx_rand = self.random_x + reg_noise_std * torch.randn_like(self.random_x).cuda()

            self.x_optimizer.zero_grad()
            self.x_scheduler.step()


            hr_pred = self.dip(self.random_x)
            # print(hr_pred.max(), hr_pred.min(), hr_pred.mean())
            # with torch.no_grad():
            #     k_pred, blur_pred = self.netG(hr_pred, hr_blur)
            
            tmp = F.conv2d(hr_pred.permute(1,0,2,3), k, padding=7).permute(1,0,2,3)

            if step%30 == 0:
                res = util.tensor2img(hr_pred.detach())
                cv2.imwrite('./test/{}.png'.format(step), res)
                res = util.tensor2img(tmp.detach())
                cv2.imwrite('./test/blur_{}.png'.format(step), res)
                # save_image(k_pred, './test/k_{}.png'.format(step),nrow=1,  normalize=True)
            # print(k_pred.shape)
            # print(blur_pred.shape)

            # lr_pred = downsample(tmp)
            # tmp =util.quantize_dip(hr_pred)
            # print('lr_pred.max(): {}, {}'.format(lr_pred.max(), lr_pred.mean()))
            # print('hr_pred.max(): {}, {}'.format(tmp.max(), tmp.mean()))

            if step < self.opt["num_iters"]//2:
                # total_loss = self.mse(tmp, hr_blur)
                total_loss = 1 - self.ssim_loss(tmp, hr_blur)
                # total_loss += 5e-5 * torch.norm(k_pred)
                # total_loss += 2e-2 * self.laplace_penalty(hr_pred)
            else:
                total_loss = self.mse(tmp, hr_blur)
                # total_loss += 5e-2 * self.laplace_penalty(hr_pred)
                # total_loss += 5e-4 * torch.norm(k_pred)

            total_loss.backward()

            self.x_optimizer.step()
            # self.k_optimizer.step()

            # debugging
            # if step % 100 == 0:
            #     print(torch.norm(k))
            #     print(f"{self.k_optimizer.param_groups[0]['lr']:.3e}")
        
        # img_lr_pred = util.Tensor2np([lr_pred.squeeze().detach().cpu().float()], 255)[0]
        # cv2.imwrite('./lr_pred.png', img_lr_pred)
        # return util.Tensor2np([hr_pred.squeeze(0).detach().cpu().float()], 255)[0]
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
        # print(checkpoint)
        if 'state_dict' in checkpoint.keys(): 
            print("YES")
            checkpoint = checkpoint['state_dict']
        
        load_func = self.SR.load_state_dict
        load_func(checkpoint)

        print('===> Loading netG module from [%s]...' % netG_path)
    
        checkpoint = torch.load(netG_path)
        if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
        load_func = self.netG.load_state_dict
        load_func(checkpoint)

    def _overlap_crop_forward(self, x, shave=10, min_size=100000, bic=None, n_GPUs=4):
        """
        chop for less memory consumption during test
        """
        n_GPUs = n_GPUs
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if bic is not None:
            bic_h_size = h_size*scale
            bic_w_size = w_size*scale
            bic_h = h*scale
            bic_w = w*scale
            
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.SR(lr_batch)

                if isinstance(sr_batch_temp, list):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp

                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output