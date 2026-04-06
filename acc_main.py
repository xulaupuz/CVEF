import os
import torch
import datetime
import torch.nn as nn
import torch.backends.cudnn as cudnn

from core import *
from misc import *
from datasets.osr_loader import CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR

import warnings
warnings.filterwarnings('ignore')


def getLoader(options):
    print("{} Preparation".format(options['dataset']))
    if 'cifar100' in options['dataset']:
        options['img_size'] = 32
        options['patch_size'] = 8  # For ViT
        Data = CIFAR100_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader = Data.train_loader, Data.test_loader

        out_Data = SVHN_OSR(known=options['unknown'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        out_loader = out_Data.test_loader
    elif 'tiny_imagenet' in options['dataset']:
        options['img_size'] = 32
        options['patch_size'] = 8  # For ViT
        Data = Tiny_ImageNet_OSR(known=options['known'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        train_loader, test_loader = Data.train_loader, Data.test_loader

        out_Data = SVHN_OSR(known=options['unknown'], batch_size=options['batch_size'], img_size=options['img_size'], options=options)
        out_loader = out_Data.test_loader

            
    options['num_known'] = Data.num_known
    return train_loader, test_loader, out_loader
 
 
def main(options):
    options['loss_keys']       = ['b1', 'b2', 'b3', 'b4', 'gate', 'divAttn', 'total']
    options['acc_keys']        = ['acc1', 'acc2', 'acc3', 'acc4', 'accGate']
    options['test_f1_keys']    = ['f1', 'f2', 'f3', 'f4', 'fGate']
    options['test_acc_keys']   = ['tacc1', 'tacc2', 'tacc3', 'tacc4', 'taccGate']
    options['test_auroc_keys'] = ['auroc1', 'auroc2', 'auroc3', 'auroc4', 'aurocGate']
    
    splits = acc_ood # new
    
    now_time = datetime.datetime.now().strftime("%m%d_%H_%M")
    log_path = './logs/osr' + '/' + options['dataset'] + '/'
    ensure_dir(log_path) # 没有会创建
    stats_log = open(log_path + "CVEFoodExp" + '_' + now_time + '.txt', 'w')

    for i in range(len(splits[options['dataset']])):
        options['item'] = i
        if options['dataset'] == 'cifar100':
            known = splits['cifar100'][i]
            unknown = splits['svhn'][i]
        else:
            known = splits['tiny_imagenet'][i]
            unknown = splits['svhn'][i]
        options.update({'known': known, 'unknown': unknown})
        print(print_options(options))
        stats_log.write(print_options(options))
        temp_result = trainLoop(options,stats_log)
        stats_log.write("SPLIT[%d|5] => Accuracy: [%.3f], AUROC: [%.3f], AUPR_IN: [%.3f], AUPR_OUT: [%.3f], F1-score: [%.3f]\n" 
                        % (i+1, temp_result[0], temp_result[1], temp_result[2], temp_result[3], temp_result[4]))
        stats_log.flush()
    stats_log.close()


def trainLoop(options,log):
    
    train_loader, test_loader, out_loader = getLoader(options)
    now_time = datetime.datetime.now().strftime("%m%d_%H_%M")
    ckpt_path = './ckpt/osr' + '/' + options['dataset'] + '/' + now_time
    ensure_dir(ckpt_path)
    model = get_model(options)

    loadpretrain.load_4b_2v(model,"C:\mlcodes\datasets\pretrainModel\imagenet21k_ViT-B_8.npz")
    # loadpretrain.load_4b_2v(model, "/HOME/scw6ceg/run/ml/datasets/pretrainModel/imagenet21k_ViT-B_8.npz")

    model = nn.DataParallel(model).cuda()

    if options['resume']:
        load_checkpoint(model, options['ckpt'])

    extractor_params = model.module.get_params(prefix='extractor')
    classifier_params = model.module.get_params(prefix='classifier')
    lr_cls = options['lr']
    lr_extractor = lr_cls
    params = [
        {'params': classifier_params, 'lr': lr_cls},
        {'params': extractor_params,  'lr': lr_extractor}
    ]
    if options['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=options['lr'], momentum=0.9, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['milestones'], gamma=options['gamma'])
    elif options['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=options['lr'], betas=(0.9,0.99), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options['milestones'], gamma=options['gamma'])
        
    entropy_loss = nn.CrossEntropyLoss().cuda()
    criterion = {'entropy': entropy_loss}

    epoch_start = 0
    bests = {'acc':0.,'aur':0.,'auin':0.,'auout':0.,'f1':0.,'dtacc':0.}
    if options['resume']:
        checkpoint_dict = load_checkpoint(model, options['ckpt'])
        epoch_start = checkpoint_dict['epoch']
        print(f'== Resuming training process from epoch {epoch_start} >')
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    for epoch in range(epoch_start, options['epoch_num']):
        lr = optimizer.param_groups[0]['lr']
        log.write(f"\nEpoch: [{epoch+1:d} | {options['epoch_num']:d}] LR: {lr:f}\n")
        print(f"\nEpoch: [{epoch+1:d} | {options['epoch_num']:d}] LR: {lr:f}")
        train_loss = train(train_loader, model, criterion, optimizer, args=options)
        if (epoch + 1) % options['test_step'] == 0:
            result_list = evaluation(model, test_loader, out_loader, **options)
            log.write(f'Acc(%): {result_list[0]:.3f}, AUROC: {result_list[1]:.5f}, AUPR_IN: {result_list[2]:.5f}, AUPR_OUT: {result_list[3]:.5f}, Macro F1-score: {result_list[4]:.5f}，DTACC: {result_list[5]:.5f}, FPR@TPR95: {result_list[6]:.5f}')
            if result_list[0] > bests['acc']: bests['acc'] = result_list[0]
            if result_list[0] > bests['aur']: bests['aur'] = result_list[1]
            if result_list[0] > bests['acc']: bests['auin'] = result_list[2]
            if result_list[0] > bests['acc']: bests['auout'] = result_list[3]
            if result_list[0] > bests['f1']: bests['f1'] = result_list[4]
            if result_list[0] > bests['dtacc']: bests['dtacc'] = result_list[5]
        scheduler.step()
        
        if (epoch + 1) % options['save_step'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, checkpoint=ckpt_path, filename=f"epoch_{epoch+1}.pth")
            if (epoch + 1) != options['save_step']:
                last_log_path=f"{ckpt_path}/epoch_{epoch+1-options['save_step']}.pth"
                if(os.path.exists(last_log_path)):
                    os.remove(last_log_path)
    
    result_list = evaluation(model, test_loader, out_loader, **options)
    log.write(f"\D-O-N-E!/ =>\nLast ACC:{result_list[0]} ,Last AUROC: {result_list[1]} ,Last F1-score: {result_list[4]}")
    print("\D-O-N-E!/ =>\nLast ACC:", result_list[0], " Last AUROC:", result_list[1]," Last F1-score:", result_list[4])
    log.write(f"Bests:{bests}")
    print("Bests:", bests)
    return result_list


if __name__ == '__main__':
    cudnn.benchmark = True
    options = get_config('osr') # 见osr.yml
    options['dataset'] = 'tiny_imagenet'#'svhn/tiny_imagenet/tiny_imagenetC/R/LSUNC/R'
    options['backbone'] = 'vit'

    set_seeding(options['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = options['gpu_ids']
    main(options)
    