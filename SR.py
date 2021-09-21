import argparse
import math
import numpy as np
import cv2
import yaml
from models.sr.blind_sr import BlindSR
import options.options as option
from data.common import random_anisotropic_gaussian_kernel, downsample, conv
from utils import util
from utils.util import img2tensor
import os
import imageio
from data import common
from torchvision.utils import save_image

# import skimage.metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
    opt = option.parse('options/test/test_catte.json')
    
    args = parser.parse_args()
    folder_path = '../SRbenchmark/HR_x4'
    for img in os.listdir(folder_path):
        image_path = os.path.join(folder_path, img)
        # Initializing mode
        model = BlindSR(opt)
        hr = common.read_img(image_path, 'img')
        k = common.random_anisotropic_gaussian_kernel(19)
        # print(k.mean(), k.max(), k.min())
        hr_tensor = common.np2Tensor([hr], 255)[0]
        print('hr.max(): {}, {}'.format(hr.max(), hr.mean()))

        input = hr_tensor.unsqueeze(0).permute(1, 0, 2, 3)
        kernel = k.unsqueeze(0)
        save_image(kernel, './test_k_GT.png',nrow=1,  normalize=True)

        hr_blur = common.conv(input, kernel, padding=19//2)
        hr_blur = hr_blur.permute(1, 0, 2, 3).float()
        print(hr_blur.mean(), hr_blur.max(), hr_blur.min())
        img_blur = util.Tensor2np([hr_blur.squeeze(0)], 255)[0]
        cv2.imwrite('./img_blur.png', cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
        lr_tensor = common.downsample(hr_blur).float()
        lr_img = util.Tensor2np([lr_tensor.squeeze(0)], 255)[0]
        cv2.imwrite('./img_lr.png', cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB))
        hr_blur = hr_blur.squeeze(0)
        res = model.SR_step(lr_tensor.cuda())
        psnr, ssim = calc_psnr(hr, res),\
                                    calc_ssim(hr, res)
        print(psnr, ssim)
        cv2.imwrite(img, res)

main()
