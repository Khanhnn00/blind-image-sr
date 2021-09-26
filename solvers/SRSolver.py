import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil
import math
from models import create_model
from .base_solver import BaseSolver
from models import init_weights
from utils import util
from torchvision.utils import make_grid
import numpy as np
from torchvision.utils import save_image

class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.opt = opt
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.k = self.Tensor()
        self.HR = self.Tensor()
        self.HR_blur = self.Tensor()
        self.HR_blur_pred = None
        self.SR = None

        self.records_SR = {'train_loss': [],
                        'lr': []
        }

        self.records_netG = {'train_loss': [],
                        'lr': []
        }
        self.model = create_model(opt)

        if self.is_train:
            self.model.train()

            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix_SR = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix_SR = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)

            
            self.criterion_pix_k = nn.L1Loss()
            self.criterion_pix_netG = nn.L1Loss()


            if self.use_gpu:
                self.criterion_pix_SR = self.criterion_pix_SR.cuda()
                self.criterion_pix_k = self.criterion_pix_k.cuda()
                self.criterion_pix_netG = self.criterion_pix_netG.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer1 = optim.Adam(self.model.module.SR.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer2 = optim.Adam(self.model.module.netG.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler1 = optim.lr_scheduler.MultiStepLR(self.optimizer1,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler2 = optim.lr_scheduler.MultiStepLR(self.optimizer2,
                                                                self.train_opt['lr_steps_netG'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()
        self.print_network()

        print('===> Solver Initialized : [%s]  || Use GPU : [%s]'%(self.__class__.__name__, self.use_gpu))
        if self.is_train:
            print("optimizer1: ", self.optimizer1)
            print("optimizer2: ", self.optimizer2)
            print("lr_scheduler1 milestones: %s   gamma: %f"%(self.scheduler1.milestones, self.scheduler1.gamma))
            print("lr_scheduler2 milestones: %s   gamma: %f"%(self.scheduler2.milestones, self.scheduler2.gamma))

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(self.model, init_type)


    def feed_data(self, batch, need_HR=True, is_train=True, which=1):
        input = batch['LR']
        self.LR.resize_(input.size()).copy_(input)

        if need_HR and is_train:
            target = batch['HR']
            blur = batch['HR_blur']
            k = batch['k']
            # print('k trong feed data {} {}'.format(k.mean(), k.max()))
            self.HR.resize_(target.size()).copy_(target)
            self.HR_blur.resize_(blur.size()).copy_(blur)
            self.k.resize_(k.size()).copy_(k)
        elif need_HR and not is_train:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)

    def feed_data_val(self, batch, which=1):
        if which==1:
            input = batch['LR']
            self.LR.resize_(input.size()).copy_(input)

            target = batch['HR_blur']
            self.HR_blur.resize_(target.size()).copy_(target)
        else:
            target = batch['HR']
            blur = batch['HR_blur']
            k = batch['k']
            # print('k trong feed data val {} {}'.format(k.mean(), k.max()))
            self.HR.resize_(target.size()).copy_(target)
            self.HR_blur.resize_(blur.size()).copy_(blur)
            self.k.resize_(k.size()).copy_(k)      

    def train_m1(self):
        self.model.module.netG.eval()

        self.model.module.SR.train()
        self.optimizer1.zero_grad()

        loss_batch = 0.0
        sub_batch_size = int(self.LR.size(0) / self.split_batch)
        for i in range(self.split_batch):
            with torch.autograd.set_detect_anomaly(True):
                loss_sbatch = 0.0
                split_LR = self.LR.narrow(0, i*sub_batch_size, sub_batch_size)
                split_HR_blur = self.HR_blur.narrow(0, i*sub_batch_size, sub_batch_size)
                output = self.model.module.SR(split_LR)
                loss_sbatch = self.criterion_pix_SR(output, split_HR_blur)

                loss_sbatch /= self.split_batch
                loss_sbatch.backward()

                loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer1.step()
            self.last_epoch_loss = loss_batch
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_batch))

        self.model.module.SR.eval()
        return loss_batch

    def normal_kl(self,mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.
        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        kl= 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)
        return kl

    def train_m2(self):
        self.model.module.SR.eval()
        self.model.module.netG.train()

        self.optimizer2.zero_grad()

        loss_batch = 0.0
        sub_batch_size = int(self.LR.size(0) / self.split_batch)
        for i in range(self.split_batch):
            with torch.autograd.set_detect_anomaly(True):
                loss_sbatch = 0.0
                split_LR = self.LR.narrow(0, i*sub_batch_size, sub_batch_size)
                split_k = self.k.narrow(0, i*sub_batch_size, sub_batch_size)
                split_HR = self.HR.narrow(0, i*sub_batch_size, sub_batch_size)
                split_HR_blur = self.HR_blur.narrow(0, i*sub_batch_size, sub_batch_size)
                with torch.no_grad():
                    HR_blur = self.model.module.SR(split_LR)
                # pred_k, pred_blur = self.model.module.netG(split_HR, HR_blur)
                pred_k, pred_blur = self.model.module.netG(split_HR, HR_blur) # k [B, ksize**2, 1, 1]
                # loss_sbatch = self.criterion_pix_k(pred_k, torch.reshape(split_k, (split_k.shape[0], -1, 1,1))) + self.criterion_pix_netG(pred_blur, split_HR_blur)
                mean1, var1 = [],[]
                mean2, var2 = [],[]
                for i in range(split_k.shape[0]):
                    mean1.append(torch.mean(pred_k[i]))
                    var1.append(torch.var(pred_k[i]))
                    mean2.append(torch.mean(split_k[i]))
                    var2.append(torch.var(split_k[i]))
                logvar1 = torch.from_numpy(np.log(var1).astype(np.float32)).cuda()
                logvar2 = torch.from_numpy(np.log(var2).astype(np.float32)).cuda()
                mean1 = torch.Tensor(mean1).cuda()
                mean2 = torch.Tensor(mean2).cuda()
                # loss_sbatch = self.criterion_pix_k(pred_k, torch.reshape(split_k, (split_k.shape[0], -1, 1,1))) + self.criterion_pix_netG(pred_blur, split_HR_blur) + self.normal_kl(mean1, logvar1, mean2, logvar2)
                loss_sbatch = self.criterion_pix_k(pred_k, torch.reshape(split_k, (split_k.shape[0], -1, 1,1))) + self.normal_kl(mean1, logvar1, mean2, logvar2)
                loss_sbatch /= self.split_batch
                loss_sbatch.backward()

                loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            self.optimizer2.step()
            self.last_epoch_loss = loss_batch
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_batch))

        self.model.module.netG.eval()
        return loss_batch

    def test(self, which):
        if which ==1:
            self.model.module.SR.eval()
            with torch.no_grad():
                forward_func = self._overlap_crop_forward if self.use_chop else self.model.forward
                SR = forward_func(self.LR, n_GPUs=2)

            self.SR = SR
            self.SR_crop = SR[:,:,:224,:224]
            self.HR_crop = self.HR_blur[:,:,:224,:224]

            self.model.module.SR.train()
            if self.is_train:
                loss_pix = self.criterion_pix_SR(self.SR, self.HR_blur)
                return loss_pix.item()
        else:
            self.model.module.netG.eval()
            with torch.no_grad():
                pred_k, pred_blur = self.model.module.netG(self.HR, self.HR_blur)
                # print('pred_k.shape: {}'.format(pred_k.shape))
                self.SR = torch.reshape(pred_k, ((pred_k.shape[0],1, 19,19)))

                self.HR_blur_pred_crop = pred_blur[:,:,:224,:224]
                self.HR_blur_crop = self.HR_blur[:,:,:224,:224]
                self.HR_blur_pred = pred_blur
            self.model.module.netG.train()
            if self.is_train:
                loss_pix = self.criterion_pix_k(self.SR, self.k) + self.criterion_pix_netG(self.HR_blur_pred, self.HR_blur)
                return loss_pix.item()


    def _forward_x8(self, x, forward_function):
        """
        self ensemble
        """
        def _transform(v, op):
            v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = self.Tensor(tfnp)

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = []
        for aug in lr_list:
            sr = forward_function(aug)
            if isinstance(sr, list):
                sr_list.append(sr[-1])
            else:
                sr_list.append(sr)

        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


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

                sr_batch_temp = self.model.module.SR(lr_batch)

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
        
    

    def save_checkpoint(self, epoch, is_best, module):
        """
        save checkpoint to experimental dir
        """
        if module == 1:
            filename = os.path.join(self.checkpoint_dir, 'SR_last_ckp.pth')
            print('===> Saving last checkpoint to [%s] ...]'%filename)
            ckp = {
                'epoch': epoch,
                'state_dict': self.model.module.SR.state_dict(),
                'optimizer': self.optimizer1.state_dict(),
                'best_pred': self.best_pred_SR,
                'best_epoch': self.best_epoch_SR,
                'records': self.records_SR
            }
        else:
            filename = os.path.join(self.checkpoint_dir, 'netG_last_ckp.pth')
            print('===> Saving last checkpoint to [%s] ...]'%filename)
            ckp = {
                'epoch': epoch,
                'state_dict': self.model.module.netG.state_dict(),
                'optimizer': self.optimizer2.state_dict(),
                'best_pred': self.best_pred_netG,
                'best_epoch': self.best_epoch_netG,
                'records': self.records_netG
            }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
            torch.save(ckp, filename.replace('last_ckp','best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp','epoch_%d_ckp'%epoch)))

            torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp'%epoch))


    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['solver']['pretrain_SR']) or not self.is_train:
            model_path_SR = self.opt['solver']['pretrainedSR_path']
            if self.opt['solver']['pretrain_netG']:
                model_path_netG = self.opt['solver']['pretrainednetG_path']
                if model_path_netG is None: raise ValueError("[Error] The 'pretrainednetG_path' does not declarate in *.json")
            if model_path_SR is None: raise ValueError("[Error] The 'pretrainedSR_path' does not declarate in *.json")
            
            print('===> Loading SR module from from {}...'.format( model_path_SR))
            if self.opt['solver']['pretrain_netG']:
                print('===> Loading netG module  from {}...'.format(model_path_netG))
            if self.is_train:
                checkpoint_SR = torch.load(model_path_SR)
                self.model.module.SR.load_state_dict(checkpoint_SR['state_dict'])
                if self.opt['solver']['pretrain_netG']:
                    checkpoint_netG = torch.load(model_path_netG)
                    self.model.module.netG.load_state_dict(checkpoint_netG['state_dict'])

                if self.opt['solver']['pretrain_SR'] == 'resume':
                    self.cur_epoch_SR = checkpoint_SR['epoch'] + 1
                    # self.optimizer1.load_state_dict(checkpoint_SR['optimizer'])
                    self.best_pred_SR = checkpoint_SR['best_pred']
                    self.best_epoch_SR = checkpoint_SR['best_epoch']
                    self.records_SR = checkpoint_SR['records']
                
                if self.opt['solver']['pretrain_netG'] == 'resume':
                    self.cur_epoch_netG = checkpoint_netG['epoch'] + 1
                    self.optimizer2.load_state_dict(checkpoint_netG['optimizer'])
                    self.best_pred_netG = checkpoint_netG['best_pred']
                    self.best_epoch_netG = checkpoint_netG['best_epoch']
                    self.records_netG = checkpoint_netG['records']
                    print(self.records_netG)

            else:
                checkpoint_SR = torch.load(model_path_SR)
                checkpoint_netG = torch.load(model_path_netG)
                if 'state_dict' in checkpoint_SR.keys(): checkpoint_SR = checkpoint_SR['state_dict']
                load_func = self.model.module.SR.load_state_dict
                load_func(checkpoint_SR)
                if 'state_dict' in checkpoint_netG.keys(): checkpoint_netG = checkpoint_netG['state_dict']
                load_func = self.model.module.netG.load_state_dict
                load_func(checkpoint_netG)


    def get_current_visual(self, need_np=True, need_HR=True, which=1):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        if which ==1:
            out_dict['LR'] = self.LR.data[0].float().cpu()
            out_dict['SR'] = self.SR.data[0].float().cpu()
            out_dict['SR_crop'] = self.SR_crop.data[0].float().cpu()
            out_dict['HR_crop'] = self.HR_crop.data[0].float().cpu()
            
            if need_np:  out_dict['LR'], out_dict['SR'], out_dict['SR_crop'] = util.Tensor2np([out_dict['LR'], out_dict['SR'], out_dict['SR_crop']],
                                                                            self.opt['rgb_range'])
            if need_HR:
                out_dict['HR'] = self.HR_blur.data[0].float().cpu()
                out_dict['HR_crop'] = self.HR_crop.data[0].float().cpu()
                if need_np: out_dict['HR'], out_dict['HR_crop'] = util.Tensor2np([out_dict['HR'], out_dict['HR_crop']],
                                                            self.opt['rgb_range'])
            return out_dict
        else:
            out_dict['LR'] = self.LR.data[0].float().cpu()
            out_dict['SR'] = self.SR.data[0].cpu()
            out_dict['HR_pred'] = self.HR_blur_pred_crop.data[0].float().cpu()
            out_dict['HR_blur'] = self.HR_blur_crop.data[0].float().cpu()
            if need_np:  
                out_dict['LR'], out_dict['HR_pred'], out_dict['HR_blur'] = util.Tensor2np([out_dict['LR'], out_dict['HR_pred'], out_dict['HR_blur']], self.opt['rgb_range'])
                out_dict['SR'] = util.Tensor2np([out_dict['SR']], -1)[0]
                                                                        
            if need_HR:
                out_dict['k'] = self.k.data[0].cpu()
                if need_np: out_dict['k'] = util.Tensor2np([out_dict['k']],
                                                            -1)[0]

            return out_dict


    def save_current_visual(self, hr_blur, hr_blur_pred):
        """
        save visual results for comparison
        """
        n_img = len(hr_blur)
        hr_blur = make_grid(hr_blur, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        hr_blur = np.transpose(hr_blur[[2, 1, 0], :, :], (1, 2, 0))
        hr_blur_pred = make_grid(hr_blur_pred, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        hr_blur_pred = np.transpose(hr_blur_pred[[2, 1, 0], :, :], (1, 2, 0))
        print('Save current visual')
        cv2.imwrite(os.path.join(self.visual_dir, './train_GT_blur.png'), hr_blur)
        cv2.imwrite(os.path.join(self.visual_dir, './train_GT_blur_pred.png'), hr_blur_pred)

    def save_img(self, epoch, iter, gt, est, hr_blur, hr_blur_pred):
        
        """
        save visual results for comparison
        """
        print('save o: {}'.format(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1))))
        gt = gt.clamp(min=0)
        est = est.clamp(min=0)
        gt_max, _ = gt.flatten(2).max(2, keepdim=True)
        gt = gt / gt_max.unsqueeze(3)
        est_max, _ = est.flatten(2).max(2, keepdim=True)
        est = est / est_max.unsqueeze(3)
        print(hr_blur.mean(), hr_blur.max(), hr_blur.min())
        print(hr_blur_pred.mean(), hr_blur_pred.max(), hr_blur_pred.min())
        save_image(gt, os.path.join(self.visual_dir, 'GT_img.png' ), nrow=10, normalize=True)
        save_image(est, os.path.join(self.visual_dir, 'SR_img.png' ), nrow=10, normalize=True)
        save_image(hr_blur, os.path.join(self.visual_dir, 'HR_blur_img.png' ), nrow=10, normalize=True)
        save_image(hr_blur_pred, os.path.join(self.visual_dir, 'HR_blur_pred_img.png'), nrow=10, normalize=True)


    def get_current_learning_rate(self, module):
        if module == 1:
            return self.optimizer1.param_groups[0]['lr']
        self.optimizer2.param_groups[0]['lr'] = 1e-4/2
        return self.optimizer2.param_groups[0]['lr']

    def update_learning_rate_SR(self, epoch):
        self.scheduler1.step(epoch)

    def update_learning_rate_netG(self, epoch):
        self.scheduler2.step(epoch)

    def get_current_log_SR(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch_SR
        log['best_pred'] = self.best_pred_SR
        log['best_epoch'] = self.best_epoch_SR
        log['records'] = self.records_SR
        return log

    def set_current_log_SR(self, log):
        self.cur_epoch_SR = log['epoch']
        self.best_pred_SR = log['best_pred']
        self.best_epoch_SR = log['best_epoch']
        self.records_SR = log['records']

    def get_current_log_netG(self):
        log_netG = OrderedDict()
        log_netG['epoch'] = self.cur_epoch_netG
        log_netG['best_pred'] = self.best_pred_netG
        log_netG['best_epoch'] = self.best_epoch_netG
        log_netG['records'] = self.records_netG
        return log_netG

    def set_current_log_netG(self, log_netG):
        self.cur_epoch_netG = log_netG['epoch']
        self.best_pred_netG = log_netG['best_pred']
        self.best_epoch_netG = log_netG['best_epoch']
        self.records_netG = log_netG['records']

    def save_current_log_SR(self):
        data = {}
        for i in self.records_SR.keys():
            data[i] = self.records_SR[i]
        data_frame = pd.DataFrame(
            data,
            index=range(1, self.cur_epoch_SR + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'SR_train_records.csv'),
                          index_label='epoch')

    def save_current_log_netG(self):
        data = {}
        for i in self.records_netG.keys():
            data[i] = self.records_netG[i]
        print('self.cur_epoch_netG: {}'.format(self.cur_epoch_netG))
        print('self.cur_epoch_SR: {}'.format(self.cur_epoch_SR))
        data_frame = pd.DataFrame(
            data,
            index=range(1, self.cur_epoch_netG + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'netG_train_records.csv'),
                          index_label='epoch')

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")