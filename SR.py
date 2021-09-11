import argparse
import math
import numpy as np
import cv2
import yaml
from models.sr.blind_sr import BlindSR
import options.options as option
from data.common import random_anisotropic_gaussian_kernel, downsample, conv
from utils.util import img2tensor
import os
# import skimage.metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calc_ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    #	parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    #	opt = option.parse(parser.parse_args().opt)
    opt = option.parse('options/train/train_IDK.json')
    args = parser.parse_args()
    image_path = '../SRbenchmark/HR_x4/bird_HR_x4.png'
    # Initializing mode
    model = BlindSR(opt)
    k = random_anisotropic_gaussian_kernel().unsqueeze(0).cuda()
    print(k.shape)
    hr_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    hr = img2tensor(hr_img).unsqueeze(0).cuda()
    blur = conv(hr.permute(1,0,2,3), k, padding=7).permute(1,0,2,3)
    lr = downsample(blur)
    res = model.SR_step(lr, hr)
    psnr, ssim = calc_psnr(hr_img, res),\
                                calc_ssim(hr_img, res)
    print(psnr, ssim)
    cv2.imwrite('./result.png', res)


main()
