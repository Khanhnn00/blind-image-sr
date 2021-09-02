from models.backbones import edsr
from models.dips import ImageDIP
from models.kernel_encoding.kernel_wizard import KernelExtractor
from sr import common

from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict

import models.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn

class IDK(nn.Module):
    def __init__(self, opt):
        super(IDK, self).__init__(opt)

        # define network and load pretrained models
        model = []
        self.DIP = ImageDIP(opt["network"]["DIP"]).to(self.device)
        self.SR = edsr.EDSR(opt["network"]["SR"]["in_channels"], opt["network"]["SR"]["out_channels"], opt["network"]["SR"]["num_features"], opt["SR"]["num_blocks"]\
            , opt["network"]["SR"]["res_scale"], opt["network"]["SR"]["upscale_factor"]).to(self.device)
        self.netG = KernelExtractor(opt["network"]["KernelExtractor"]).to(self.device)
        self.is_train = opt["is_train"]

        self.scale = opt["network"]["SR"]["upscale_factor"]
        model.append(self.DIP)
        model.append(self.SR)
        model.append(self.netG)

        self.model = nn.Sequential(*model)


    def forward(self, lr=None, zx=None, hr=None):
        if self.is_train:
            output = self._forward_train(lr, hr)
        else:
            output = self._forward_test(lr, zx)
        return output


    def _forward_train(self, lr, hr):
        assert lr is not None
        assert hr is not None
        hr_b_pred = self.SR(lr)
        k_pred = self.netG(hr_b_pred, hr)
        return hr_b_pred, k_pred

        
    def _forward_test(self, lr, zx):
        hr_b_pred = self.SR(lr)
        hr_pred = self.DIP(zx)
        k_pred = self.netG(hr_b_pred, hr_pred)
        lr_pred = common.degradation(hr_pred, k_pred, self.scale)
        return lr_pred