import functools

import models.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import ResidualBlock_noBN, ResnetBlock
from models.backbones.unet_parts import UnetSkipConnectionBlock
from models.backbones import edsr


# The function F in the paper
class KernelExtractor(nn.Module):
    def __init__(self, opt):
        super(KernelExtractor, self).__init__()

        nf = opt["nf"]
        self.kernel_dim = opt["kernel_dim"]
        self.use_sharp = opt["use_sharp"]

        # Blur estimator
        norm_layer = arch_util.get_norm_layer(opt["norm"])
        n_blocks = opt["n_blocks"]
        padding_type = opt["padding_type"]
        use_dropout = opt["use_dropout"]
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        input_nc = nf * 2 if self.use_sharp else nf
        output_nc = self.kernel_dim * 2 if self.use_vae else self.kernel_dim

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(nf),
            nn.ReLU(True),
        ]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            inc = min(nf * mult, output_nc)
            ouc = min(nf * mult * 2, output_nc)
            model += [
                nn.Conv2d(inc, ouc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(nf * mult * 2),
                nn.ReLU(True),
            ]

        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    output_nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        self.model = nn.Sequential(*model)

    def forward(self, sharp, blur):
        output = self.model(torch.cat((sharp, blur), dim=1))
        return output, torch.zeros_like(output).cuda()


class StupidCatte(nn.Module):
    def __init__(self, opt):
        super(StupidCatte, self).__init__()
        lrelu = nn.LeakyReLU(negative_slope=0.1)
        front_RBs = opt["front_RBs"]
        back_RBs = opt["back_RBs"]
        num_image_channels = opt["input_nc"]
        nf = opt["nf"]

        # Features extraction
        resBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        feature_extractor = []

        feature_extractor.append(nn.Conv2d(num_image_channels, nf, 3, 1, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)
        feature_extractor.append(nn.Conv2d(nf, nf, 3, 2, 1, bias=True))
        feature_extractor.append(lrelu)

        for i in range(front_RBs):
            feature_extractor.append(resBlock_noBN_f())

        self.feature_extractor = nn.Sequential(*feature_extractor)

        # Kernel extractor
        self.kernel_extractor = KernelExtractor(opt)

        # SR module
        self.SR = edsr.EDSR(opt["network"]["SR"]["in_channels"], opt["network"]["SR"]["out_channels"], opt["network"]["SR"]["num_features"], opt["network"]["SR"]["num_blocks"]\
            , opt["network"]["SR"]["res_scale"], opt["network"]["SR"]["upscale_factor"])

    def downsample(img):
        return F.interpolate(img, scale_factor=(1/4, 1/4), mode='bicubic')

    def SR_step(self, lr):
        return self.SR(lr)

    def forward(self, lr, hr):
        hr_blur = self.SR_step(lr)
        hr = self.feature_extractor(hr)
        hr_blur = self.feature_extractor(hr_blur)

        output = self.kernel_extractor(hr, hr_blur)

        lr_pred = []
        for i in range(output.shape[0]):
            tmp = F.conv2d(hr[i].unsqueeze(0).permute(1,0,2,3), output[i].unsqueeze(0), padding=7).permute(1,0,2,3)
            tmp = self.downsample(tmp)
            lr_pred.append(tmp)
        lr_pred = torch.cat(lr_pred, dim=0).float()
        return output, lr_pred