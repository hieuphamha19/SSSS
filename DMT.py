"""
Dual-Teacher
Copyright (c) 2023-present NAVER Cloud Corp.
Distributed under NVIDIA Source Code License for SegFormer

References:
  SegFormer: https://github.com/NVlabs/SegFormer
"""

import os
import time
import argparse
from datetime import datetime
from itertools import cycle

import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from copy import deepcopy

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

from Datasets.create_dataset import get_dataset_without_full_label, Dataset, StrongWeakAugment
from Models.DeepLabV3Plus.modeling import deeplabv3plus_resnet101
from Utils.utils import DotDict, fix_all_seed
from Losses.DMT_loss import compute_cutmix, compute_classmix, compute_ic, ClassMixLoss


def sigmoid_rampup(current, rampup_length):
    """
    Compute the sigmoid ramp-up value for consistency loss weighting.
    
    Args:
        current (float): Current epoch.
        rampup_length (int): Total epochs for ramp-up.
        
    Returns:
        float: Sigmoid ramp-up value between 0 and 1.
    """
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, args):
    """
    Compute the current consistency weight.
    
    Args:
        epoch (int): Current epoch.
        args: The argument parser object with consistency settings.
        
    Returns:
        float: Current consistency weight.
    """
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def create_model(ema=False):
    """
    Create a DeepLabV3+ model and move it to GPU.
    
    Args:
        ema (bool): If True, detach parameters for EMA.
        
    Returns:
        torch.nn.Module: The model.
    """
    model = deeplabv3plus_resnet101(num_classes=3, output_stride=8, pretrained_backbone=True).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def update_ema(model_teacher, model, alpha_teacher, iteration):
    """
    Update the teacher model parameters using Exponential Moving Average.
    
    Args:
        model_teacher (torch.nn.Module): Teacher model.
        model (torch.nn.Module): Student model.
        alpha_teacher (float): EMA decay rate.
        iteration (int): Current iteration number.
    """
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(model_teacher.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param.data[:] + (1 - alpha_teacher) * param.data[:]


def validate_model(model, val_loader, criterion):
    """
    Validate the model performance on the validation dataset.
    
    Args:
        model (torch.nn.Module): The model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        
    Returns:
        dict: Aggregated metrics dictionary.
    """
    model.eval()
    metrics = {'dice': 0, 'iou': 0, 'hd': 0, 'loss': 0}
    num_val = 0

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0)

    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]

        with torch.no_grad():
            output = torch.softmax(model(img), dim=1)
            loss = criterion(output, label)

            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)

            # Convert labels to one-hot if needed.
            if len(label.shape) != 4:  
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)
            else:
                labels_onehot = label

            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_metric(y_pred=preds_onehot, y=labels_onehot)

            metrics['loss'] = (metrics['loss'] * num_val + loss.item() * batch_len) / (num_val + batch_len)
            num_val += batch_len

            val_loop.set_postfix({'Loss': f"{loss.item():.4f}"})

    metrics['dice'] = dice_metric.aggregate().item()
    metrics['iou'] = iou_metric.aggregate().item()
    metrics['hd'] = hd_metric.aggregate().item()

    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()

    return metrics


def train_val(config, model, model_teacher, model_teacher2, train_loader, val_loader,
              criterion, criterion_u, cm_loss_fn, best_model_dir, file_log):
    """
    Training and validation function with Dual Teacher strategy.
    
    Args:
        config: Experiment configuration.
        model: Student model.
        model_teacher/model_teacher2: Teacher models with EMA.
        train_loader (dict): Dictionary of labeled and unlabeled DataLoaders.
        val_loader (DataLoader): Validation DataLoader.
        criterion: Supervised loss function.
        criterion_u: Unsupervised loss function.
        cm_loss_fn: ClassMix loss function.
        best_model_dir (str): Path to save the best model.
        file_log (file object): Log file handle.
    
    Returns:
        torch.nn.Module: Trained model.
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs)
    train_dice = DiceMetric(include_background=True, reduction="mean")
    
    max_dice = -float('inf')
    best_epoch = 0
    iter_num = 0

    # Save initial model.
    torch.save(model.state_dict(), best_model_dir)
    
    for epoch in range(config.train.num_epochs):
        start = time.time()
        model.train()
        train_metrics = {'dice': 0, 'loss': 0}
        num_train = 0

        # Select teacher and mixing strategy.
        if epoch % 2 == 0:
            ema_model = model_teacher
            do_cut_mix = True
            do_class_mix = False
        else:
            ema_model = model_teacher2
            do_cut_mix = False
            do_class_mix = True
        
        source_dataset = zip(cycle(train_loader['l_loader']), train_loader['u_loader'])
        train_loop = tqdm(source_dataset, desc=f'Epoch {epoch} Training', leave=False)
        train_dice.reset()

        for idx, (batch, batch_u) in enumerate(train_loop):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            image_u = batch_u['img_w'].cuda().float()
            label_u = batch['label'].cuda().float()
            sup_batch_len = img.shape[0]

            # Use strong augmentation provided in the batch.
            image_u_strong = batch_u['img_s'].cuda().float()

            if do_class_mix:
                loss = compute_classmix(
                    criterion=criterion,
                    cm_loss_fn=cm_loss_fn,
                    model=model,
                    ema_model=ema_model,
                    imgs=img,
                    labels=label,
                    unsup_imgs=image_u,
                    image_u_strong=image_u_strong,
                    threshold=0.95
                )
            elif do_cut_mix:
                loss = compute_cutmix(
                    imgs=img,
                    labels=label,
                    criterion=criterion,
                    model=model,
                    ema_model=ema_model,
                    image_u=image_u,
                    threshold=0.95
                )

            # Consistency loss.
            loss_dc = compute_ic(
                model=model,
                ema_model=ema_model,
                image_u=image_u,
                image_u_strong=image_u_strong,
                criterion_u=criterion_u,
                label_u=label_u,
                threshold=0.95
            )
            # Total loss.
            total_loss = loss + loss_dc * 0.2
            # For reference: you can also apply a consistency weight.
            # consistency_weight = get_current_consistency_weight(epoch, config)
            # total_loss = loss + loss_dc * consistency_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema(model_teacher=ema_model, model=model, alpha_teacher=0.99, iteration=iter_num)

            with torch.no_grad():
                output = torch.softmax(model(img), dim=1)
                output_onehot = torch.zeros_like(output)
                output_onehot.scatter_(1, output.argmax(dim=1, keepdim=True), 1)
                train_dice(y_pred=output_onehot, y=label)
                train_metrics['loss'] = (train_metrics['loss'] * num_train + total_loss.item() * sup_batch_len) / (num_train + sup_batch_len)
                num_train += sup_batch_len

            train_loop.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Dice': f"{train_dice.aggregate().item():.4f}"
            })
            iter_num += 1

            if config.debug:
                break

        train_metrics['dice'] = train_dice.aggregate().item()
        val_metrics = validate_model(model, val_loader, criterion)
        current_dice = val_metrics['dice']

        if current_dice > max_dice:
            max_dice = current_dice
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            message = f'New best epoch {epoch}! Dice: {current_dice:.4f}'
            print(message)
            file_log.write(message + '\n')
            file_log.flush()

        scheduler.step()
        time_elapsed = time.time() - start
        print(f'Epoch {epoch} completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

        if config.debug:
            break

    print(f'Training completed. Best epoch: {best_epoch}')
    return model


def test(config, model, best_model_dir, test_loader, criterion, test_results_dir, file_log):
    """
    Test the model using the saved best checkpoint.
    
    Args:
        config: Experiment configuration.
        model (torch.nn.Module): Model instance.
        best_model_dir (str): Path to the saved best model.
        test_loader (DataLoader): Test DataLoader.
        criterion: Loss function.
        test_results_dir (str): Directory to store test results.
        file_log (file object): Log file handle.
    """
    model.load_state_dict(torch.load(best_model_dir))
    metrics = validate_model(model, test_loader, criterion)

    results_str = (
        f"Test Results:\n"
        f"Loss: {metrics['loss']:.4f}\n"
        f"Dice: {metrics['dice']:.4f}\n"
        f"IoU: {metrics['iou']:.4f}\n"
        f"HD: {metrics['hd']:.4f}"
    )

    with open(test_results_dir, 'w') as f:
        f.write(results_str)

    print('=' * 80)
    print(results_str)
    print('=' * 80)

    file_log.write('\n' + '=' * 80 + '\n')
    file_log.write(results_str + '\n')
    file_log.write('=' * 80 + '\n')
    file_log.flush()


def main(config, best_model_dir, test_results_dir, file_log, args):
    """
    Main training function.
    
    Args:
        config: Experiment configuration.
        best_model_dir (str): Path to save the best model.
        test_results_dir (str): Path to store test results.
        file_log (file object): Log file handle.
        args: Parsed command-line arguments.
    """
    dataset = get_dataset_without_full_label(
        config, 
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment
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

    model = create_model()
    model_teacher = create_model(ema=True)
    model_teacher2 = create_model(ema=True)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/1e6:.2f}M total parameters')
    print(f'{total_trainable_params/1e6:.2f}M trainable parameters')

    criterion = DiceCELoss(
        include_background=True,
        to_onehot_y=False,
        softmax=True,
        reduction='mean'
    ).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    cm_loss_fn = ClassMixLoss(weight=None, reduction='none', ignore_index=255)

    model = train_val(config, model, model_teacher, model_teacher2, train_loader,
                      val_loader, criterion, criterion_u, cm_loss_fn,
                      best_model_dir, file_log)
    
    test(config, model, best_model_dir, test_loader,
         criterion, test_results_dir, file_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with FixMatch')
    parser.add_argument('--exp', type=str, default='tmp')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--consistency', type=float, default=1)
    parser.add_argument('--consistency_rampup', type=float, default=75.0)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load and update config.
    with open(args.config_yml, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method'] = args.adapt_method
    config['model_adapt']['num_domains'] = args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    # Setup CUDA and seeds.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])
    
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")
    
    store_config = config.copy()
    config = DotDict(config)
    
    # Train for each fold.
    for fold in [1, 2, 3, 4, 5]:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        exp_dir = os.path.join(config.data.save_folder, args.exp, f"fold{fold}")
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = os.path.join(exp_dir, 'best.pth')
        test_results_dir = os.path.join(exp_dir, 'test_results.txt')
        
        # Save configuration if not in debug.
        if not config.debug:
            with open(os.path.join(exp_dir, 'exp_config.yml'), 'w') as f:
                yaml.dump(store_config, f)
        
        with open(os.path.join(exp_dir, 'log.txt'), 'w') as file_log:
            main(config, best_model_dir, test_results_dir, file_log, args)
        
        torch.cuda.empty_cache()

