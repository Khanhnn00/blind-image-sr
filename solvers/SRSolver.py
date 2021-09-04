import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil

from models import create_model
from .base_solver import BaseSolver
from models import init_weights
from utils import util

class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.opt = opt
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.k = self.Tensor()
        self.HR = self.Tensor()
        self.HR_blur = self.Tensor()
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

            # set cl_loss

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer1 = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer2 = optim.Adam(self.model.parameters(),
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
                                                                self.train_opt['lr_steps'],
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
            self.HR.resize_(target.size()).copy_(target)
            self.HR_blur.resize_(blur.size()).copy_(blur)
            self.k.resize_(k.size()).copy_(k)
        elif need_HR and not is_train:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)

    def feed_data_val(self, batch, which=1):
        if which ==1:
            input = batch['LR']
            self.LR.resize_(input.size()).copy_(input)

            target = batch['HR_blur']
            self.HR_blur.resize_(target.size()).copy_(target)
        else:
            target = batch['HR']
            blur = batch['HR_blur']
            k = batch['k']
            self.HR.resize_(target.size()).copy_(target)
            self.HR_blur.resize_(target.size()).copy_(blur)
            self.k.resize_(target.size()).copy_(k)
        
        

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
                loss_sbatch = self.criterion_pix(output, split_HR_blur)

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
                with torch.no_grad():
                    HR_blur = self.model.module.SR(split_LR)
                output = self.model.module.netG(split_HR, HR_blur)
                loss_sbatch = self.criterion_pix(output, split_k)

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

            self.model.module.SR.train()
            if self.is_train:
                loss_pix = self.criterion_pix(self.SR, self.HR_blur)
                return loss_pix.item()
        else:
            self.model.module.netG.eval()
            with torch.no_grad():
                pred_k = self.model.module.netG(self.HR, self.HR_blur)
                self.SR = pred_k
            self.model.module.netG.train()
            if self.is_train:
                loss_pix = self.criterion_pix(self.SR, self.k)
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
                'best_pred': self.best_pred1,
                'best_epoch': self.best_epoch1,
                'records': self.records1
            }
        else:
            filename = os.path.join(self.checkpoint_dir, 'netG_last_ckp.pth')
            print('===> Saving last checkpoint to [%s] ...]'%filename)
            ckp = {
                'epoch': epoch,
                'state_dict': self.model.module.netG.state_dict(),
                'optimizer': self.optimizer2.state_dict(),
                'best_pred': self.best_pred2,
                'best_epoch': self.best_epoch2,
                'records': self.records2
            }
        # else:
        #     filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        #     print('===> Saving last checkpoint to [%s] ...]'%filename)
        #     ckp = {
        #         'epoch': epoch,
        #         'state_dict': self.model.module.netG.state_dict(),
        #         'optimizer': self.optimizer2.state_dict(),
        #         'best_pred': self.best_pred2,
        #         'best_epoch': self.best_epoch2,
        #         'records': self.records2
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
        if (self.is_train and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None: raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['solver']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.best_pred = checkpoint['best_pred']
                    self.best_epoch = checkpoint['best_epoch']
                    self.records = checkpoint['records']

            else:
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
                load_func = self.model.module.load_state_dict if isinstance(self.model, nn.DataParallel) \
                    else self.model.module.load_state_dict
                load_func(checkpoint)


    def get_current_visual(self, need_np=True, need_HR=True, which=1):
        """
        return LR SR (HR) images
        """
        out_dict = OrderedDict()
        if which ==1:
            out_dict['LR'] = self.LR.data[0].float().cpu()
            out_dict['SR'] = self.SR.data[0].float().cpu()

            
            if need_np:  out_dict['LR'], out_dict['SR'] = util.Tensor2np([out_dict['LR'], out_dict['SR']],
                                                                            self.opt['rgb_range'])
            if need_HR:
                out_dict['HR'] = self.HR_blur.data[0].float().cpu()
                if need_np: out_dict['HR'] = util.Tensor2np([out_dict['HR']],
                                                            self.opt['rgb_range'])[0]
            return out_dict
        else:
            out_dict['LR'] = self.LR.data[0].float().cpu()
            out_dict['SR'] = self.SR.data[0].float().cpu()
            if need_np:  out_dict['LR'], out_dict['SR'] = util.Tensor2np([out_dict['LR'], out_dict['SR']],
                                                                            self.opt['rgb_range'])
            if need_HR:
                out_dict['k'] = self.k.data[0].float().cpu()
                if need_np: out_dict['k'] = util.Tensor2np([out_dict['k']],
                                                            -1)[0]
            return out_dict



    def save_current_visual(self, epoch, iter):
        """
        save visual results for comparison
        """
        if epoch % self.save_vis_step == 0:
            visuals_list = []
            visuals = self.get_current_visual(need_np=False)
            visuals_list.extend([util.quantize(visuals['HR'].squeeze(0), self.opt['rgb_range']),
                                 util.quantize(visuals['SR'].squeeze(0), self.opt['rgb_range'])])
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            misc.imsave(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1)),
                        visual_images)


    def get_current_learning_rate(self, module):
        if module == 1:
            return self.optimizer1.param_groups[0]['lr']
        return self.optimizer2.param_groups[0]['lr']

    def update_learning_rate1(self, epoch):
        self.scheduler1.step(epoch)

    def update_learning_rate2(self, epoch):
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
        for i in self.records.keys():
            data[i] = self.records_SR[i]
        data_frame = pd.DataFrame(
            data,
            index=range(1, self.cur_epoch + 1)
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'SR_train_records.csv'),
                          index_label='epoch')

    def save_current_log_netG(self):
        data = {}
        for i in self.records.keys():
            data[i] = self.records_netG[i]
        data_frame = pd.DataFrame(
            data,
            index=range(1, self.cur_epoch + 1)
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