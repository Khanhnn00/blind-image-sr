import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K', 'DF2K']


####################
# Files & IO
####################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images


def _get_paths_from_binary(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_binary_file(fname):
                binary_path = os.path.join(dirpath, fname)
                files.append(binary_path)
    assert files, '[%s] has no valid binary file' % path
    return files


def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        elif data_type == 'npy':
            if dataroot.find('_npy') < 0 :
                old_dir = dataroot
                dataroot = dataroot + '_npy'
                if not os.path.exists(dataroot):
                    print('===> Creating binary files in [%s]' % dataroot)
                    os.makedirs(dataroot)
                    img_paths = sorted(_get_paths_from_images(old_dir))
                    path_bar = tqdm(img_paths)
                    for v in path_bar:
                        img = imageio.imread(v, pilmode='RGB')
                        ext = os.path.splitext(os.path.basename(v))[-1]
                        name_sep = os.path.basename(v.replace(ext, '.npy'))
                        np.save(os.path.join(dataroot, name_sep), img)
                else:
                    print('===> Binary files already exists in [%s]. Skip binary files generation.' % dataroot)

            paths = sorted(_get_paths_from_binary(dataroot))

        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths


def find_benchmark(dataroot):
    bm_list = [dataroot.find(bm)>=0 for bm in BENCHMARK]
    if not sum(bm_list) == 0:
        bm_idx = bm_list.index(True)
        bm_name = BENCHMARK[bm_idx]
    else:
        bm_name = 'MyImage'
    return bm_name


def read_img(path, data_type):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if data_type == 'img':
        img = imageio.imread(path, pilmode='RGB')
    elif data_type.find('npy') >= 0:
        img = np.load(path)
    else:
        raise NotImplementedError

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return modcrop(img, 4)
####################
#for BD degradation#

def get_imgs(path):
    # print(path)
    hr = np.load(path)
    # print(hr.shape)
    hr = modcrop(hr, scale=4)
    hr_x = cv2.GaussianBlur(hr,(7,7),1.6).clip(0, 255)
    lr = misc.imresize(hr_x, 1 / 4, interp='bicubic')
    # print(type(lr))
    return lr.clip(0, 255).astype(np.uint8), hr_x.clip(0, 255).astype(np.uint8), hr.clip(0, 255).astype(np.uint8)

def get_patch_hrx(img_in, img_x, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]

    ip = patch_size

    if ih == oh:
        tp = ip
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = ix, iy
    else:
        tp = ip * scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_x = img_x[ty:ty + tp, tx:tx + tp, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_x, img_tar

####################
# image processing
# process on numpy image
####################
def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        # print(type(img))
        #if img.shape[2] == 3: # for opencv imread
        #    img = img[:, :, [2, 1, 0]]
        # print(img.shape)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose.copy()).float()
        tensor.mul_(rgb_range / 255.)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def k2tensor(k):
    np_transpose = np.ascontiguousarray(k.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose.copy()).float()

    return tensor

def conv(inp, k, padding):
    #inp: B C H W
    #k: B 1 15 15
    return F.conv2d(inp, k, padding=padding)

def downsample(img):
    return F.interpolate(img, scale_factor=(1/4, 1/4), mode='bicubic')

def get_patch(img_tar, patch_size, scale):
    # print(type(img_tar))
    oh, ow = img_tar.shape[:2]


    ip = patch_size

    tp = ip * scale
    tx = random.randrange(0, ow - tp + 1)
    ty = random.randrange(0, oh - tp + 1)

    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    # print(type(img_tar))
    return img_tar

def get_patch_lrx(img_in, img_inx, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]

    ip = patch_size

    if ih == oh:
        tp = ip
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = ix, iy
    else:
        tp = ip * scale
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_inx = img_inx[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_inx, img_tar

def quantize_to_1(img, rgb_range):
    if rgb_range != -1:
        pixel_range = 1. / rgb_range
        # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
        return img.mul(pixel_range).clamp(0, 1).round()
    else:
        return img


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def inv_covariance_matrix(sig_x, sig_y, theta):
    # sig_x : x-direction standard deviation
    # sig_x : y-direction standard deviation
    # theta : rotation angle
    D_inv = np.array([[1/(sig_x ** 2), 0.], [0., 1/(sig_y ** 2)]])  # inverse of diagonal matrix D
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # eigenvector matrix
    inv_cov = np.dot(U, np.dot(D_inv, U.T))  # inverse of covariance matrix
    return inv_cov

def anisotropic_gaussian_kernel(width, inv_cov):
    # width : kernel size of anisotropic gaussian filter
    ax = np.arange(-width // 2 + 1., width // 2 + 1.)
    # avoid shift
    if width % 2 == 0:
        ax = ax - 0.5
    xx, yy = np.meshgrid(ax, ax)
    xy = np.stack([xx, yy], axis=2)
    # pdf of bivariate gaussian distribution with the covariance matrix
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inv_cov) * xy, 2))
    kernel = kernel / np.sum(kernel)
    return kernel


def random_anisotropic_gaussian_kernel(width=15, sig_min=0.2, sig_max=4.0):
    # width : kernel size of anisotropic gaussian filter
    # sig_min : minimum of standard deviation
    # sig_max : maximum of standard deviation
    sig_x = np.random.random() * (sig_max - sig_min) + sig_min
    sig_y = np.random.random() * (sig_max - sig_min) + sig_min
    theta = np.random.random() * 3.141/2.
    inv_cov = inv_covariance_matrix(sig_x, sig_y, theta)
    kernel = anisotropic_gaussian_kernel(width, inv_cov)
    kernel = kernel.astype(np.float32)
    kernel = np.expand_dims(kernel, axis=0)
    return torch.from_numpy(kernel)

def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [%d].' % img.ndim)
    return img
