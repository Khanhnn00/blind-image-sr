import argparse, random
from tqdm import tqdm

import torch
import skimage.metrics
import numpy as np
import options.options as option
from utils import util
import os
from solvers import create_solver_cate
from data import create_dataloader
from data import create_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train(train_loader, train_set, val_set, epoch, NUM_EPOCH, solver, solver_log, loader_list, opt, which):
    # print(solver_log)
    train_loss_list = []
    with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
        for iter, batch in enumerate(train_loader):
            solver.feed_data(batch)
            if which ==1:
                iter_loss = solver.train_m1()
            else:
                iter_loss = solver.train_m2()
            batch_size = batch['LR'].size(0)
            train_loss_list.append(iter_loss*batch_size)
            t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
            t.update()

        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate(which))

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set)))

        print('===> Validating...',)

        val_loader = loader_list[which-1]
        val_loss_list = []
        psnr_list = []
        ssim_list = []
        visual = []
        visual_gt = []
        hr_blur = []
        hr_blur_pred = []
        for iter, batch in enumerate(val_loader):
            solver.feed_data_val(batch, which=which)
            iter_loss = solver.test(which=which)
            val_loss_list.append(iter_loss)

            # calculate evaluation metrics
            visuals = solver.get_current_visual(which=which)
            if which ==1:
                psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=4, test_Y=False)
                hr_blur_pred.append(np.expand_dims(visuals['SR_crop'], axis=0).transpose(0,3,1,2))
                hr_blur.append(np.expand_dims(visuals['HR_crop'], axis=0).transpose(0,3,1,2))
                
            else:
                visual_gt.append(np.expand_dims(visuals['k'], axis=0).transpose(0,3,1,2))
                visual.append(np.expand_dims(visuals['SR'], axis=0).transpose(0,3,1,2))
                hr_blur_pred.append(np.expand_dims(visuals['HR_pred'], axis=0).transpose(0,3,1,2))
                hr_blur.append(np.expand_dims(visuals['HR_blur'], axis=0).transpose(0,3,1,2))
                psnr, ssim = skimage.metrics.peak_signal_noise_ratio(visuals['k'], visuals['SR']),\
                                skimage.metrics.structural_similarity(visuals['k'], visuals['SR'], multichannel=True)
                
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        if which ==1:
            hr_blur = np.concatenate(hr_blur, axis=0)
            hr_blur_pred = np.concatenate(hr_blur_pred, axis=0)
            hr_blur = hr_blur[:100,:,:,:]
            hr_blur_pred = hr_blur_pred[:100,:,:,:]
            hr_blur = torch.from_numpy(hr_blur)
            hr_blur_pred = torch.from_numpy(hr_blur_pred)
            if opt["save_image"]:
                solver.save_current_visual(hr_blur, hr_blur_pred)
        else:
            visual_gt = np.concatenate(visual_gt, axis=0)
            hr_blur = np.concatenate(hr_blur, axis=0)
            hr_blur_pred = np.concatenate(hr_blur_pred, axis=0)
            visual = np.concatenate(visual, axis=0)
            visual = torch.from_numpy(visual)
            visual_gt = torch.from_numpy(visual_gt)
            hr_blur = torch.from_numpy(hr_blur).float()
            hr_blur_pred = torch.from_numpy(hr_blur_pred).float()
            hr_blur = hr_blur[:100,:,:,:]
            hr_blur_pred = hr_blur_pred[:100,:,:,:]

            if opt["save_image"]:
                solver.save_img(epoch, iter, visual_gt, visual, hr_blur, hr_blur_pred)
        if 'val_loss' not in solver_log['records']:
            solver_log['records']['val_loss']= []
        if 'psnr' not in solver_log['records']:
            solver_log['records']['psnr'] = []
        if 'ssim' not in solver_log['records']:
            solver_log['records']['ssim'] = []
        
        solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
        solver_log['records']['psnr'].append(sum(psnr_list)/len(psnr_list))
        solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))

    # record the best epoch
    epoch_is_best = False
    # print(solver_log['records']['psnr_5'])
    if solver_log['best_pred'] < sum(psnr_list)/len(psnr_list):
        solver_log['best_pred'] = sum(psnr_list)/len(psnr_list)
        epoch_is_best = True
        solver_log['best_epoch'] = epoch


    print("[%s] PSNR: %.2f   SSIM: %.4f   Loss: %.6f   Best PSNR: %.2f in Epoch: [%d]" % (val_set.name(),
                                                                                            sum(psnr_list)/len(psnr_list),
                                                                                            sum(ssim_list)/len(ssim_list),
                                                                                            sum(val_loss_list)/len(val_loss_list),
                                                                                            solver_log['best_pred'],
                                                                                            solver_log['best_epoch']))
                                                                                                                                                                    
    if which == 1:
        solver.set_current_log_SR(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best, which)
        solver.save_current_log_SR()

        # update lr
        solver.update_learning_rate_SR(epoch)
    else:
        solver.set_current_log_netG(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best, which)
        solver.save_current_log_netG()

        # update lr
        solver.update_learning_rate_netG(epoch)

    print('===> Finished !')

def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    #	parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    #	opt = option.parse(parser.parse_args().opt)
    opt = option.parse('options/train/train_catte.json')


    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]"%seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    loader_list =[]
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase.find('val') == 0:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            loader_list.append(val_loader)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))
        
        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    solver = create_solver_cate(opt)
    scale = opt['scale']
    model_name = opt['network']['which_model'].upper()
    print(model_name)

    print('===> Start Train')
    print("==================================================")

    solver_log_SR = solver.get_current_log_SR()
    solver_log_netG = solver.get_current_log_netG()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    if solver.opt['solver']['netG_only']:
        start_epoch = solver_log_netG['epoch']
    else:
        start_epoch = solver_log_SR['epoch']
    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)"%(model_name, scale, start_epoch, NUM_EPOCH))

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        if solver.opt['solver']['netG_only']:
            print('\n===> Training Epoch: [%d/%d] netG module ...  Learning Rate: %f'%(epoch,
                                                                        NUM_EPOCH,
                                                                        solver.get_current_learning_rate(2)))        
            solver_log_netG['epoch'] = epoch
            train(train_loader, train_set, val_set, epoch, NUM_EPOCH, solver, solver_log_netG, loader_list, opt, 2)
        
        else:
            print('\n===> Training Epoch: [%d/%d] SR module ...  Learning Rate: %f'%(epoch,
                                                                        NUM_EPOCH,
                                                                        solver.get_current_learning_rate(1)))
        
            solver_log_SR['epoch'] = epoch
            train(train_loader, train_set, val_set, epoch, NUM_EPOCH, solver, solver_log_SR, loader_list, opt, 1)
    


if __name__ == '__main__':
    main()