import os
import time
import argparse
from datetime import datetime
import cv2

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle
from monai.losses import *
from monai.metrics import *

from Datasets.create_dataset import *
from Models.DeepLabV3Plus.modeling import *
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed

def create_model(ema=False):
    """
    Create a new model instance with EMA support.
    """
    model = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Update teacher model parameters using exponential moving average.
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def sigmoid_rampup(current, rampup_length):
    """
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    """Calculate the consistency weight for the current epoch."""
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def main(config):
    """
    Main training function for Mean Teacher method.
    """
    # Setup datasets and dataloaders
    dataset = get_dataset_without_full_label(
        config,
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=SkinDataset2,
        ulb_dataset=StrongWeakAugment4
    )
    
    l_train_loader = DataLoader(
        dataset['lb_dataset'],
        batch_size=config.train.l_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    u_train_loader = DataLoader(
        dataset['ulb_dataset'],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset['val_dataset'],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True
    )
    
    train_loader = {'l_loader': l_train_loader, 'u_loader': u_train_loader}
    print(f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}")

    # Create student and teacher models
    model = create_model()
    ema_model = create_model(ema=True)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    model = model.cuda()
    ema_model = ema_model.cuda()

    # Setup loss functions
    criterion_sup = GeneralizedDiceFocalLoss(
        softmax=True,
        to_onehot_y=False,
        include_background=True
    ).cuda()
    
    criterion_cons = nn.MSELoss().cuda()

    # Train and test
    best_model = train_val(config, model, ema_model, train_loader, val_loader, 
                          criterion_sup, criterion_cons)
    test(config, best_model, best_model_dir, test_loader, criterion_sup)

def train_val(config, model, ema_model, train_loader, val_loader, criterion_sup, criterion_cons):
    """
    Training and validation function for Mean Teacher method.
    """
    # Setup optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs, eta_min=1e-6)

    # Initialize metrics
    train_dice = DiceMetric(include_background=True, reduction="mean")
    
    # Training loop
    max_dice = -float('inf')
    best_epoch = 0
    global_step = 0
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        
        # Training phase
        model.train()
        ema_model.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        
        for batch_idx, (batch_l, batch_u) in enumerate(train_loop):
            # Get labeled data
            img_l = batch_l['image'].cuda().float()
            label_l = batch_l['label'].cuda().float()
            
            # Get unlabeled data (single augmentation)
            img_u = batch_u['img_w'].cuda().float()
            
            # Forward passes
            outputs_l = model(img_l)
            outputs_u = model(img_u)
            
            with torch.no_grad():
                teacher_outputs_u = ema_model(img_u)
            
            # Calculate supervised loss
            sup_loss = criterion_sup(outputs_l, label_l)
            
            # Calculate consistency loss
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = criterion_cons(outputs_u.softmax(dim=1), 
                                           teacher_outputs_u.softmax(dim=1))
            
            # Total loss
            loss = sup_loss + consistency_weight * consistency_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update teacher model
            global_step += 1
            update_ema_variables(model, ema_model, args.ema_decay, global_step)
            
            # Calculate metrics
            with torch.no_grad():
                output_onehot = torch.zeros_like(outputs_l)
                output_onehot.scatter_(1, outputs_l.argmax(dim=1, keepdim=True), 1)
                
                train_dice(y_pred=output_onehot, y=label_l)
                
                train_metrics['loss'] = (train_metrics['loss'] * num_train + loss.item() * img_l.shape[0]) / (num_train + img_l.shape[0])
                num_train += img_l.shape[0]
            
            # Update progress bar
            train_loop.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}"
            })
            
            if config.debug:
                break
        
        # Get final training metrics
        train_metrics['dice'] = train_dice.aggregate().item()
        
        # Validation phase
        val_metrics = validate_model(ema_model, val_loader, criterion_sup)
        
        # Save best model
        if val_metrics['dice'] > max_dice:
            max_dice = val_metrics['dice']
            best_epoch = epoch
            torch.save(ema_model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch}! '
                      f'Dice: {val_metrics["dice"]:.4f}')
            print(message)
            file_log.write(message + '\n')
            file_log.flush()
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch time
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print('='*80)
        
        if config.debug:
            break
    
    print(f'Training completed. Best epoch: {best_epoch}')
    return ema_model

def validate_model(model, val_loader, criterion):
    """
    Validate model using MONAI metrics.
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0
    
    # Initialize metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)
            
            # Convert predictions to one-hot
            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)
            
            # Convert labels to one-hot if needed
            if len(label.shape) == 4:
                labels_onehot = label
            else:
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)
            
            # Update metrics
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)
            
            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * img.shape[0]) / (num_val + img.shape[0])
            num_val += img.shape[0]
    
    # Aggregate metrics
    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()
    
    # Reset metrics
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
    return metrics

def test(config, model, model_dir, test_loader, criterion):
    """
    Test the model on test set.
    """
    model.load_state_dict(torch.load(model_dir))
    metrics = validate_model(model, test_loader, criterion)
    
    results_str = (f"Test Results:\n"
                  f"Loss: {metrics['loss']:.4f}\n"
                  f"Dice: {metrics['dice']:.4f}\n"
                  f"IoU: {metrics['iou']:.4f}\n"
                  f"HD: {metrics['hd']:.4f}")
    
    with open(test_results_dir, 'w') as f:
        f.write(results_str)
    
    print('='*80)
    print(results_str)
    print('='*80)
    
    file_log.write('\n' + '='*80 + '\n')
    file_log.write(results_str + '\n')
    file_log.write('='*80 + '\n')
    file_log.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mean Teacher Training')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    args = parser.parse_args()
    
    # Load and update config
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['data']['name'] = args.dataset
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    # Setup CUDA and seeds
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])
    
    # Print configuration
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")
    
    store_config = config
    config = DotDict(config)
    
    # Train each fold
    for fold in [1,2,3,4,5]:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Setup directories
        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f'{exp_dir}/best.pth'
        test_results_dir = f'{exp_dir}/test_results.txt'
        
        # Save config
        if not config.debug:
            yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
        
        # Train fold
        with open(f'{exp_dir}/log.txt', 'w') as file_log:
            main(config)
        
        torch.cuda.empty_cache()
