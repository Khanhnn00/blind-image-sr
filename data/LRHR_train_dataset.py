import torch.utils.data as data
import torch
import torch.nn.functional as F
from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return (self.opt['dataroot_HR'].split('/')[-1])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        print(opt)
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR= None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 1

        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])

        assert self.paths_HR, '[Error] HR paths are empty.'


    def __getitem__(self, idx):
        hr, hr_path = self._load_file(idx)
        if self.train:
            hr = self._get_patch(hr)
            k = common.random_anisotropic_gaussian_kernel(self.opt['kernel_size'])
        hr_tensor = common.np2Tensor([hr], self.opt['rgb_range'])[0]
        input = hr_tensor.unsqueeze(0)

        input = hr_tensor.unsqueeze(0).permute(1, 0, 2, 3)
        # input = F.pad(input, pad=(self.opt['kernel_size'] // 2, self.opt['kernel_size']// 2, self.opt['kernel_size']// 2, self.opt['kernel_size']// 2),
        #                 mode='circular')
        kernel = k.unsqueeze(0)

        hr_blur = common.conv(input, kernel, padding=self.opt['kernel_size']//2)
        # print(hr_blur.shape)
        hr_blur = hr_blur.permute(1, 0, 2, 3)
        lr_tensor = common.downsample(hr_blur)
        hr_blur = hr_blur.squeeze(0)
        lr_tensor = lr_tensor.squeeze(0)
        # print('lr.shape: {}'.format(lr_tensor.shape))
        # print('k.shape: {}'.format(k.shape))
        # print('hr_blur.shape: {}'.format(hr_blur.shape))
        # print('hr.shape: {}'.format(hr_tensor.shape))
        return {'LR': lr_tensor, 'k': k, 'HR_blur': hr_blur, 'HR': hr_tensor, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        hr_path = self.paths_HR[idx]
        hr = common.read_img(hr_path, self.opt['data_type'])

        return hr, hr_path


    def _get_patch(self, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        hr = common.get_patch(
            hr, LR_size, self.scale)
        hr = common.augment([hr])[0]
        # print(type(hr))

        return hr                                                                                                                                                                                                                                                                                                                         