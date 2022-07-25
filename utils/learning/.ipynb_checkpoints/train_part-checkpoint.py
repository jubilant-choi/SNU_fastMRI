import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss, save_exp_result
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet
from utils.model.unet_advanced import Unet as newUnet

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    with tqdm(data_loader, unit="batch") as tepoch:
        for iter, data in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}")
            
            input, target, maximum, _, _ = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            output = model(input)
            loss = loss_type(output, target, maximum)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    #         if iter % args.report_interval == 0:
    #             print(
    #                 f'Epoch = [{epoch+1:3d}/{args.num_epochs:3d}] '
    #                 f'Iter = [{iter+1:4d}/{len(data_loader):4d}] '
    #                 f'Loss = {loss.item():.4g} '
    #                 f'Time = {time.perf_counter() - start_iter:.4f}s',
    #             )
    #         start_iter = time.perf_counter()
            tepoch.set_postfix(loss=loss.item())
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, scheduler):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
        
    scheduler.step(metric_loss)  
    
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / f'{args.exp_name+"_epoch"+str(epoch)}.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / f'{args.exp_name}_best.pt')
        
    if epoch > 1:
        os.remove(args.exp_dir / f'{args.exp_name+"_epoch"+str(epoch-1)}.pt')

def select_model(args):
    net_name = args.net_name.name
    if net_name == 'test_Unet':
        model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    elif net_name == 'newUnet':
        model = newUnet(in_chans = args.in_chans, out_chans = args.out_chans)
    else:
        raise Exception("Invalid Model was given as an argument :", net_name)
    
    return model
        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(' - Current .cuda device: ', torch.cuda.current_device())
    
    model = select_model(args)  
    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, min_lr=1e-9)
    
    if args.load == '':
        start_epoch = 0 
        best_val_loss = 1. 
    else:
        print(f'\n*** Load Checkpoint for f{args.load} ***')
        checkpoint = torch.load(args.exp_dir / args.load)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Start Epoch = {start_epoch+1}, best validation loss = {best_val_loss:0.5f}")
        print(f"Previous learning rate was {checkpoint['optimizer']['param_groups'][0]['lr']}\n")
    
    result = {}
    result['train_losses'] = []
    result['val_losses'] = []
    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    for epoch in tqdm(range(start_epoch, start_epoch+args.num_epochs)):
        print(f'Epoch #{epoch+1:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, scheduler)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print('\n'
            f'* Epoch = [{epoch+1:4d}/{args.num_epochs:4d}] | Loss (Train/Val) = {train_loss:.4g} / {val_loss:.4g}'
            f"| Time(Train/Val) = {train_time:.4f}s / {val_time:.4f}s | Learning rate = {optimizer.param_groups[0]['lr']}",
        )
        
        result['train_losses'].append(train_loss)
        result['val_losses'].append(val_loss)
        
        if is_new_best:
            print("*** NewRecord ***")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f' - ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            
        
        save_exp_result(save_dir=args.json_dir, setting=deepcopy(vars(args)), result=deepcopy(result), args.load)
