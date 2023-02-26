from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Crack
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torchvision import transforms as T

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from utils.utils import increment_path
import cv2

import wandb


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/cracktree260',help="path to Dataset")
    # Model Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='fcn_denseunetv2', choices=available_models, help='model name')

    # other
    parser.add_argument("--dataset", type=str, default='crack',choices=['voc', 'cityscapes','crack'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,help="num classes (default: None)")
    
    
    parser.add_argument("--separable_conv", action='store_true', default=False, help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3, help="epoch number (default: 30k)")
    parser.add_argument("--total_epochs", type=int, default=4000, help="epoch number (default: 400)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'], help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False, help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=400)
    parser.add_argument("--ckpt", default='', type=str, help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--loss_type", type=str, default='binary', choices=['cross_entropy', 'focal_loss', 'binary','weighted'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10, help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=5, help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False, help="download datasets")
    parser.add_argument("--project", default = 'checkpoints', help = 'save to directory')
    parser.add_argument("--name", default = 'exp', help = 'save to directory/name')
    parser.add_argument("--earlystop", action='store_false',default = True, help = 'early stop')

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012', choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False, help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570', help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main', help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8, help='number of samples for visualization (default: 8)')
    parser.add_argument("--wandb", type=str, default='Segmentation_binary', help='env for visdom')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( (480,720)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)


    if opts.dataset == 'crack':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            # et.ExtRandomRotation(180),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( (480,720)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Crack(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Crack(root=opts.data_root,
                             split='val', transform=val_transform)

    return train_dst, val_dst


def validate(model, loader, device, metrics, save_path, deepcrack=False):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if deepcrack:
                outputs = model(images)[:,-1,:,:][:,None,:,:].sigmoid()
            else:
                outputs = model(images).sigmoid()
            preds = outputs.detach().max(dim=1)[0].cpu().numpy()

            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            
            if i == 0:
                out = (255 - preds[2,:,:] * 255).astype('uint8')
                cv2.imwrite(save_path, out)

        score = metrics.get_results()
    return score


def main():
  
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'crack':
        opts.num_classes = 2

    wandb.init(project = opts.wandb)
    opts.save_dir = str(increment_path(Path(opts.project) / opts.name, exist_ok = False))

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    using_deepcrack = False

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    class_weight = torch.tensor([1,10]).float().to(device)
    if opts.model[0:9] == 'deepcrack':
        criterion = utils.DeepCrackLoss()
        using_deepcrack = True
    elif opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss( ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(weight= class_weight,ignore_index=255, reduction='mean')
    elif opts.loss_type == 'binary':
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    elif opts.loss_type == 'weighted':
        criterion = utils.WeightedLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    unc_interval = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
  
    interval_loss = 0
    # transform for updating ground truth
    transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])


    while True:  # cur_epochs < opts.total_epochs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for idx, (images, labels) in tqdm(enumerate(train_loader)):
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)[:,None,:,:]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)
        interval_loss = interval_loss / (idx + 1)
        print("Epoch %d, Itrs %d/%d, Loss=%f" %
            (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))  

        if (cur_epochs) % opts.val_interval == 0:
            unc_interval += 1
            save_ckpt(opts.save_dir +'/latest_%s_%s_os%d.pth' %
                        (opts.model, opts.dataset, opts.output_stride))
            print("validation...")
            model.eval()

            val_score = validate(
                model=model, loader=val_loader, device=device, metrics=metrics,
                save_path = opts.save_dir + '/epochs%05d.jpg' % cur_epochs, deepcrack =using_deepcrack)

            wandb.log({'Train_loss': interval_loss, 'Val_ACC': val_score['Overall Acc'], 'Val_IoU': val_score['Mean IoU'],'Val_IoU(Crack)': val_score['Class IoU'][1], 
                        'Precision(crack)': val_score['Precision'][1], 'Recall(crack)':val_score['Recall'][1], 'F1(crack)':val_score['F1'][1]})

            print(metrics.to_str(val_score))

            if val_score['Mean IoU'] > best_score:  # save best model
                unc_interval = 0
                best_score = val_score['Mean IoU']
                save_ckpt(opts.save_dir +'/best_%s_%s_os%d.pth' %
                            (opts.model, opts.dataset, opts.output_stride))

            model.train()
            scheduler.step()

            if opts.earlystop and unc_interval >= 100:
                print('Early stop')
                return

            if cur_epochs >= opts.total_epochs:
                return
        
        interval_loss = 0.0


if __name__ == '__main__':
    main()
