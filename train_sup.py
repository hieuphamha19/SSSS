'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 1 --adapt_method False --num_domains 1 --dataset drive --k_fold 2 > 2MedFormer_drive.out 2>&1 &

'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from Datasets.create_dataset import get_dataset, SkinDataset2, get_dataset_without_full_label
from Utils.losses import dice_loss, hausdorff_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Models.Transformer.SwinUnet import SwinUnet
from Models.resunet import ResUNet
from Models.unetCCT import UNet

from Utils.metrics import calc_metrics, calc_hd, calc_dice, calc_iou

torch.cuda.empty_cache()

loss_weights = [0.3, 0.5, 0.2]  

def main(config):
    
    # dataset = get_dataset(config, img_size=config.data.img_size, 
    #                                             supervised_ratio=config.data.supervised_ratio, 
    #                                             train_aug=config.data.train_aug,
    #                                             k=config.fold,
    #                                             lb_dataset=SkinDataset2)
    dataset = get_dataset_without_full_label(config, img_size=config.data.img_size,
                                                train_aug=config.data.train_aug,
                                                k=config.fold,
                                                lb_dataset=SkinDataset2)
    

    train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize, 
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False, 
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    test_loader = torch.utils.data.DataLoader(dataset['val_dataset'], 
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers, 
                                                pin_memory=True,
                                                drop_last=False,
                                                persistent_workers=True,
                                                prefetch_factor=2)
    print(len(train_loader), len(dataset['lb_dataset']))

    
    # model = SwinUnet(config.data.img_size, num_classes=3,window_size=8)
    model = UNet(in_chns=3, class_num=3)


    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    # from thop import profile
    # input = torch.randn(1,3,224,224)
    # flops, params = profile(model, (input,))
    # print(f"total flops : {flops/1e9} G")

    # test model
    # x = torch.randn(5,3,224,224)
    # y = model(x)
    # print(y.shape)

    model = model.cuda()
    
    criterion = [
        nn.CrossEntropyLoss(), 
        lambda pred, target: dice_loss(torch.softmax(pred, dim=1), target),
        lambda pred, target: hausdorff_loss(torch.softmax(pred, dim=1), target)
    ]


    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
        test(config, model, best_model_dir, test_loader, criterion)



# =======================================================================================================
def train_val(config, model, train_loader, val_loader, criterion):
    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_score = -float('inf')  # Track combined score
    best_epoch = 0
    
    # Weights for combining metrics (có thể điều chỉnh)
    w_dice = 0.5  # Weight cho Dice score 
    w_hd = 0.5    # Weight cho HD score
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model.train()
        dice_train_sum = 0
        iou_train_sum = 0
        loss_train_sum = 0
        num_train = 0

        # Add tqdm progress bar for training
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch} Training', leave=False)
        for batch in train_loop:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]
            
            output = model(img)
            output = torch.softmax(output, dim=1)
            
            # calculate loss
            assert (output.shape == label.shape)
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            
            # Weighted sum of losses
            loss = sum(w * l for w, l in zip(loss_weights, losses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train_sum += loss.item() * batch_len
            
            # calculate metrics
            with torch.no_grad():
                pred = output.argmax(dim=1).cpu().numpy()
                label = label.argmax(dim=1).cpu().numpy()
                assert (pred.shape == label.shape)
                dice_train = calc_dice(pred, label)
                iou_train = calc_iou(pred, label)
                
                dice_train_sum += dice_train * batch_len
                iou_train_sum += iou_train * batch_len
                
                # Update progress bar with Loss and Dice (only once)
                train_loop.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice_train:.4f}'
                })
            
            iter = epoch * len(train_loader) + train_loop.n
                
            file_log.write('Epoch {}, iter {}, Dice Sup Loss: {}, BCE Sup Loss: {}\n'.format(
                epoch, iter + 1, round(losses[1].item(), 5), round(losses[0].item(), 5)
            ))
            file_log.flush()
            
            num_train += batch_len
            if config.debug: break
            
            # Remove duplicate progress bar update here

        # print
        file_log.write('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}\n'.format(
            epoch, iter, 
            round(loss_train_sum / num_train, 5), 
            round(dice_train_sum / num_train, 4), 
            round(iou_train_sum / num_train, 4)))
        file_log.flush()
        print('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}'.format(
            epoch, len(train_loader), 
            round(loss_train_sum / num_train, 5), 
            round(dice_train_sum / num_train, 4), 
            round(iou_train_sum / num_train, 4)))
            
        # -----------------------------------------------------------------
        # validate
        # ----------------------------------------------------------------
        model.eval()
        
        dice_val_sum = 0
        iou_val_sum = 0
        loss_val_sum = 0
        hd_val_sum = 0  # Initialize HD sum
        num_val = 0

        # Add tqdm progress bar for validation
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch} Validation', leave=False)
        for batch in val_loop:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model(img)
                output = torch.softmax(output, dim=1)

                # calculate loss
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_val_sum += sum(losses)*batch_len

                # calculate metrics
                pred = output.argmax(dim=1).cpu().numpy()
                label = label.argmax(dim=1).cpu().numpy()
                
                # Calculate Dice & IoU
                dice_val = calc_dice(pred, label)
                iou_val = calc_iou(pred, label)
                
                dice_val_sum += dice_val * batch_len
                iou_val_sum += iou_val * batch_len
                
                # Calculate HD
                hd = calc_hd(pred, label)
                hd_val_sum += hd * batch_len
                
                # Update progress bar
                val_loop.set_postfix({
                    'Loss': f'{(sum(losses)).item():.4f}',
                    'Dice': f'{dice_val:.4f}',
                    'HD': f'{hd:.4f}'
                })
                
                num_val += batch_len
                if config.debug: break

        # Calculate epoch metrics
        loss_val_epoch = loss_val_sum/num_val
        dice_val_epoch = dice_val_sum/num_val
        iou_val_epoch = iou_val_sum/num_val
        hd_val_epoch = hd_val_sum/num_val  # Calculate average HD

        # Calculate combined score
        hd_norm = np.exp(-hd_val_epoch/100)  # Use hd_val_epoch instead of hd_val_sum
        combined_score = w_dice * dice_val_epoch + w_hd * hd_norm
        
                # Log validation metrics
        file_log.write(f'Epoch {epoch}, Validation || '
                      f'Loss: {loss_val_epoch:.4f}, '
                      f'Dice: {dice_val_epoch:.4f}, '
                      f'HD: {hd_val_epoch:.4f}, '
                      f'Combined Score: {combined_score:.4f}, '
                      f'IOU: {iou_val_epoch:.4f}\n')
        file_log.flush()
        print(f'Epoch {epoch}, Validation || Loss: {loss_val_epoch:.4f}, '
              f'Dice: {dice_val_epoch:.4f}, HD: {hd_val_epoch:.4f}, '
              f'IOU: {iou_val_epoch:.4f}')
        
        # Save model if combined score improves
        if combined_score > max_score:
            max_score = combined_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)
            
            message = (f'New best epoch {epoch} '
                      f'Combined score improved to {combined_score:.4f} ==========\n'
                      f'Dice: {dice_val_epoch:.4f}, HD: {hd_val_epoch:.4f} ==========')
            
            file_log.write(message + '\n')
            file_log.flush()
            print(message)
        


        # scheduler step, record lr
        scheduler.step()

        end = time.time()
        time_elapsed = end-start
        file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
        file_log.flush()
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return
    
    file_log.write('Complete training ---------------------------------------------------- \n The best epoch is {}\n'.format(best_epoch))
    file_log.flush()
    
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    return 




# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    dice_test_sum = 0
    iou_test_sum = 0
    loss_test_sum = 0
    hd_test_sum = 0  # Initialize HD sum
    num_test = 0

    # Add tqdm progress bar for testing
    test_loop = tqdm(test_loader, desc='Testing', leave=True)
    for batch in test_loop:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        batch_len = img.shape[0]
            
        with torch.no_grad():
            output = model(img)
            output = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            # Tính loss
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            loss_test_sum += sum(losses) * batch_len
            
            # Tính metrics
            pred = pred.cpu().numpy()
            label = label.argmax(dim=1).cpu().numpy()
            
            dice_test = calc_dice(pred, label)
            iou_test = calc_iou(pred, label)
            
            dice_test_sum += dice_test * batch_len
            iou_test_sum += iou_test * batch_len

            # Calculate HD
            hd = calc_hd(pred, label)
            hd_test_sum += hd * batch_len
            
            # Keep only one progress bar update
            test_loop.set_postfix({
                'Loss': f'{(sum(losses)).item():.4f}',
                'Dice': f'{dice_test:.4f}',
                'HD': f'{hd:.4f}'
            })

            num_test += batch_len
            
            if config.debug: break

    # Calculate average metrics
    loss_test_epoch = loss_test_sum/num_test
    dice_test_epoch = dice_test_sum/num_test
    iou_test_epoch = iou_test_sum/num_test
    hd_test_epoch = hd_test_sum/num_test  # Calculate average HD

    # Save results to file with all metrics
    with open(test_results_dir, 'w') as f:
        f.write(f'Loss: {round(loss_test_epoch.item(),4)}, '
                f'Dice: {round(dice_test_epoch,4)}, '
                f'IOU: {round(iou_test_epoch,4)}, '
                f'HD: {round(hd_test_epoch,4)}')

    # Print and log results with all metrics
    print('========================================================================================')
    print(f'Test || Loss: {round(loss_test_epoch.item(),4)}, '
          f'Dice: {round(dice_test_epoch,4)}, '
          f'IOU: {round(iou_test_epoch,4)}, '
          f'HD: {round(hd_test_epoch,4)}')
    
    file_log.write('========================================================================================\n')
    file_log.write(f'Test || Loss: {round(loss_test_epoch.item(),4)}, '
                   f'Dice: {round(dice_test_epoch,4)}, '
                   f'IOU: {round(iou_test_epoch,4)}, '
                   f'HD: {round(hd_test_epoch,4)}\n')
    file_log.flush()
 
    return




if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str,default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    folds_to_train = [1, 2, 3, 4, 5]
    
    for fold in folds_to_train:
        print(f"\n=== Training Fold {fold} ===")
        config['fold'] = fold
        
        # Update paths for each fold
        # exp_dir = '{}/{}_{}/fold{}'.format(config.data.save_folder, args.exp, config['data']['supervised_ratio'], fold)
        exp_dir = '{}/{}/fold{}'.format(config.data.save_folder, args.exp, fold)
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = '{}/best.pth'.format(exp_dir)
        test_results_dir = '{}/test_results.txt'.format(exp_dir)

        # Store yml file for each fold
        if config.debug == False:
            yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
            
        file_log = open('{}/log.txt'.format(exp_dir), 'w')
        
        # Train the model for this fold
        main(config)
        
        # Close the log file
        file_log.close()
        
        # Clear GPU memory between folds
        torch.cuda.empty_cache()