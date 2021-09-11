import argparse

import cv2
import yaml
from models.sr.blind_sr import BlindSR
import options.options as option
from data.common import random_anisotropic_gaussian_kernel, downsample, conv
from utils.util import img2tensor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
    hr = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    hr = img2tensor(hr).unsqueeze(0).cuda()
    blur = conv(hr.permute(1,0,2,3), k, padding=7).permute(1,0,2,3)
    lr = downsample(blur)
    res = model.SR_step(lr, hr)

    cv2.imwrite('./result.png', res)


main()
