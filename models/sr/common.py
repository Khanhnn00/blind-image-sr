  
import os
import random
import torch
import torch.nn.functional as F
import cv2

def degradation(img, kernel, scale_factor,device=torch.device('cuda')):


    input = F.pad(img, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[0] // 2),
                mode='circular')
    kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

    # blur
    output = F.conv2d(input, kernel)
    output = output.permute(2, 3, 0, 1).squeeze(3).cpu().numpy()

    #downsample
    h, w = output.shape[1]//4, output.shape[0]//4
    output = cv2.resize(output, (h, w), cv2.INTER_CUBIC)

    return torch.fromoutput