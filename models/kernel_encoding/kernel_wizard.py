import functools
import torch.nn.functional as F
import models.arch_util as arch_util
import torch
import torch.nn as nn
from models.backbones.resnet import ResidualBlock_noBN, ResnetBlock
from models.backbones.unet_parts import UnetSkipConnectionBlock


# The function F in the paper

class EncBlock(nn.Module):
    def __init__(self, input_nc, nf, padding_type, norm_layer, use_dropout, use_bias):
        super(EncBlock, self).__init__()
        head = [
            nn.Conv2d(input_nc, nf, kernel_size=3, padding=1, stride=1),
            ResnetBlock(
                    nf,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                ),
            ResnetBlock(
                    nf,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                ),
            nn.ReLU(True),
        ]
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*head)
    
    def forward(self,x):
        x = self.model(x)

        out = self.max(x)
        return x, out

class DecBlock(nn.Module):
    def __init__(self, input_nc, nf, padding_type, norm_layer, use_dropout, use_bias):
        super(DecBlock, self).__init__()
        head = [
            nn.ConvTranspose2d(input_nc, nf, kernel_size=4, padding=1, stride=2),
        ]
        tail = [
            nn.Conv2d(nf*2, nf, kernel_size=3, padding=1, stride=1),
            ResnetBlock(
                    nf,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                ),
            ResnetBlock(
                    nf,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                ),
            nn.ReLU(True)
        ]
        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*tail)
    
    def forward(self, x, enc):
        x = self.head(x)
        x = torch.cat((x, enc), dim=1)
        return self.tail(x)

class KernelExtractor(nn.Module):
    def __init__(self, opt):
        super(KernelExtractor, self).__init__()
        print(opt)
        nf = opt["nf"]
        self.kernel_dim = opt["kernel_dim"]
        self.use_sharp = opt["use_sharp"]
        self.use_vae = opt["use_vae"]

        # Blur estimator
        norm_layer = arch_util.get_norm_layer(opt["norm"])
        padding_type = opt["padding_type"]
        use_dropout = opt["use_dropout"]
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input_nc = opt['in_channels'] * 2 if self.use_sharp else opt['in_channels']
        output_nc = opt['out_channels'] * 2 if self.use_vae else opt['out_channels']
        
        self.enc1 = EncBlock(self.input_nc, nf, padding_type=padding_type, norm_layer=norm_layer, \
            use_bias=use_bias, use_dropout=use_dropout)
        self.enc2 = EncBlock(nf, nf*2, padding_type=padding_type, norm_layer=norm_layer, \
            use_bias=use_bias, use_dropout=use_dropout)
        self.conv1 = [
            nn.Conv2d(nf*2, nf*4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True)
        ]
        for i in range(4):

            self.conv1+= [
                nn.Conv2d(nf*4, nf*4, kernel_size=3, padding=1, stride=1),
                nn.ReLU(True)
            ]
        self.conv1 = nn.Sequential(*self.conv1)
        self.dec1 = DecBlock(nf*4, nf*2, padding_type, norm_layer, use_dropout, use_bias)
        self.dec2 = DecBlock(nf*2, nf, padding_type, norm_layer, use_dropout, use_bias)
        self.conv2 = [
            nn.Conv2d(nf, nf, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(nf, 15*15, kernel_size=3, padding=1, stride=1),
        ]
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, sharp, blur):
        if self.input_nc == 6:
            inp = torch.cat((sharp, blur), dim=1)
        else:
            inp = blur
        # print(inp.shape)
        x1_, x1 = self.enc1(inp)
        x2_, x2 = self.enc2(x1)
        x3 = self.conv1(x2)
        x4 = self.dec1(x3, x2_)
        x5 = self.dec2(x4, x1_)
        out = self.conv2(x5)
        k = torch.reshape(F.adaptive_avg_pool2d(out, (1, 1)), (out.shape[0],1,15,15))
        blur = []
        for i in range(out.shape[0]):
            tmp = F.conv2d(sharp[i].unsqueeze(0).permute(1,0,2,3), k[i].unsqueeze(0), padding=7).permute(1,0,2,3)
            blur.append(tmp)
        blur = torch.cat(blur, dim=0).float()

        return k, blur




