from tqdm import tqdm
import network
from network.backbone.deepcrack import deepcrack
import utils
import os
import argparse
import numpy as np
from pathlib import Path
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Crack
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from torchvision import transforms as T
from glob import glob

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2


"""This evaluation code is used for crack segmentation specifically."""

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", default = './datasets/data/mixed_crack', type=str, help="path to image directory")
    parser.add_argument("--dataset", type=str, default='crack', choices=['voc', 'cityscapes', 'crack'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='fcn_unet', choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False, help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results", action='store_true', help="always be false, only for using 'validate' code purpose")
    parser.add_argument("--save_val_results_to", default='test_results/unet(non_weak_supervision)', help="save segmentation results to the specified dir")
    parser.add_argument("--crop_val", action='store_true', default=False, help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=(480,720))
    parser.add_argument("--relaxation",action='store_false', help= 'relax 2 pixels for precision' )

    
    parser.add_argument("--ckpt", default='checkpoints/exp_unet(non_weak)/best_fcn_unet_crack_os16.pth', type=str,  help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    return parser

# validate
def validate(model, loader, device, metrics, tv, deepcrack = False):

    px, py = np.linspace(0, 1, 1000), []  # for plotting

    eps = 1e-16
    confusion_matrix_tv = np.zeros((tv.shape[0],4))
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

            confusion_matrix_tv += metrics.update_tv(targets, preds, tv)

        p = confusion_matrix_tv[:,3] / (confusion_matrix_tv[:,3] + confusion_matrix_tv[:, 1] + eps)
        r = confusion_matrix_tv[:,3] / (confusion_matrix_tv[:,3] + confusion_matrix_tv[:, 2] + eps)
        f1 = 2*p*r / (p + r + eps)
        ap,mpre, mrec = compute_ap(r, p)
        py = np.interp(px, mpre, mrec ) 
    
    return p, r, f1, ap, px, py

def validate_relax(model, loader, device, metrics, tv, deepcrack = False):

    px, py = np.linspace(0, 1, 1000), []  # for plotting

    eps = 1e-16
    confusion_matrix_tv = np.zeros((tv.shape[0],4))
    confusion_matrix_tv_buff = np.zeros((tv.shape[0],4))

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            buff = nn.MaxPool2d(5, stride = 1, padding = 2)
            labels_buff = buff(labels.type(torch.float32))

            if deepcrack:
                outputs = model(images)[:,-1,:,:][:,None,:,:].sigmoid()
            else:
                outputs = model(images).sigmoid()

            preds = outputs.detach().max(dim=1)[0].cpu().numpy()
            targets = labels.cpu().numpy()
            targets_buff = labels_buff.cpu().numpy()

            confusion_matrix_tv += metrics.update_tv_r(targets, preds, tv)
            confusion_matrix_tv_buff += metrics.update_tv(targets_buff, preds, tv)

        p = confusion_matrix_tv_buff[:,3] / (confusion_matrix_tv_buff[:,3] + confusion_matrix_tv_buff[:, 1] + eps)
        r = confusion_matrix_tv[:,3] / (confusion_matrix_tv[:,3] + confusion_matrix_tv[:, 2] + eps)
        f1 = 2*p*r / (p + r + eps)
        ap,mpre, mrec = compute_ap(r, p)
        py = np.interp(px, mpre, mrec ) 
    
    return p, r, f1, ap, px, py

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([1.0],recall, [0.0]))
    mpre = np.concatenate(([0.0], precision, [1.0]))

    # Compute the precision envelope
    mrec = np.flip(np.maximum.accumulate(np.flip(mrec)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mpre,mrec), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=3, color='blue', label='Crack %.3f AP' % ap)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 2
    decode_fn = Crack.decode_target
    using_deepcrack = False

    metrics = StreamSegMetrics(opts.num_classes)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    if opts.model[0:9] == 'deepcrack':
        using_deepcrack = True

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input + '/images/val', '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    transform_1 = T.Compose([
                #T.Resize(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    
    transform_2 = et.ExtCompose([
            #et.ExtResize( opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    # create validation dataloader
    val_dst = Crack(root = opts.input, split='val',transform=transform_2)

    val_loader = data.DataLoader(val_dst, batch_size = opts.val_batch_size, shuffle=False, num_workers=2)

    #print('Updating validation ground truth')
    #val_dst.update_gt(model, transform = transform_1, device = device)
    
    with torch.no_grad():
        model = model.eval()

        tv = np.linspace(0.05,0.95,10)

        if opts.relaxation:
            p, r, f1, ap, px, py = validate_relax(model = model, loader= val_loader, device= device, metrics = metrics, tv = tv, deepcrack= using_deepcrack)

        else:
            p, r, f1, ap, px, py = validate(model = model, loader= val_loader, device= device, metrics = metrics, tv = tv, deepcrack = using_deepcrack)

        plot_pr_curve(px, py, ap, save_dir=os.path.join(opts.save_val_results_to,'pr_curve.png'), names=['crack'])

        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform_1(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)

            if using_deepcrack:
                pred = model(img)[:,-1,:,:][:,None,:,:].sigmoid().detach().max(dim=1)[0].cpu().numpy()
            else:
                pred = model(img).sigmoid().detach().max(dim=1)[0].cpu().numpy()
            
            #pred = model(img).sigmoid().max(1)[0].cpu().numpy() # HW

            if opts.save_val_results_to:
                out = (255 - pred[0,:,:]* 255).astype('uint8')
                cv2.imwrite(os.path.join(opts.save_val_results_to, img_name+'.png'), out)

if __name__ == '__main__':
    main()