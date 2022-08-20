import os
import copy
import time
import gc
import shutil
import requests
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss, save_exp_result
from utils.common.loss_function import SSIMLoss
from utils.model.testUnet import Unet as testUnet
from utils.model.unet import Unet, UnetPP
from utils.model.varnet import VarNet
from utils.model.adaptive_varnet import AdaptiveVarNet


def center_mask(reduce_size=30):
    mask = np.ones((384,384),dtype=np.int8)
    r = 384//2
    center = np.array((r,r))

    for x in range(384):
        for y in range(384):
            if np.linalg.norm(np.array((x,y))-center) >= r-reduce_size:
                mask[x][y] = 0
    return torch.tensor(mask,device='cuda')


def train_epoch(args, epoch, model, data_loader, optimizer, scaler, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    cmask = center_mask()
    
    with tqdm(data_loader, unit="batch", mininterval=5) as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for iter, data in enumerate(tepoch):
            
            with torch.autocast(device_type='cuda'):
                if args.input_key != 'kspace':
                    _, input, target, maximum, _, _ = data
    #                 input = input.cuda(non_blocking=True)
                    output = model(input.cuda(non_blocking=True))

                elif 'VarNet' == args.net_name.name:
                    mask, kspace, target, maximum, _, _ = data
    #                 mask = mask.cuda(non_blocking=True)
    #                 kspace = kspace.cuda(non_blocking=True)
                    output = model(kspace.cuda(non_blocking=True), mask.cuda(non_blocking=True))

    #             output = output * cmask
    #             target = target.cuda(non_blocking=True)
    #             maximum = maximum.cuda(non_blocking=True)

            loss = loss_type(output * cmask, target.cuda(non_blocking=True), maximum.cuda(non_blocking=True)) if args.loss == 'ssim' else loss_type(output, target)
        
            optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            tepoch.set_postfix(loss=loss.item())
            
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, scheduler):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict) if args.input_key != 'kspace' else None
    cmask = center_mask()
    start = time.perf_counter()

    with torch.no_grad():
        if args.input_key == 'kspace' and args.net_name.name == 'VarNet':
            for iter, data in enumerate(data_loader):
                mask, kspace, target, _, fnames, slices = data
#                 kspace = kspace.cuda(non_blocking=True)
#                 mask = mask.cuda(non_blocking=True)
                output = model(kspace.cuda(non_blocking=True), mask.cuda(non_blocking=True))
                output = output * cmask
                
                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    targets[fnames[i]][int(slices[i])] = target[i].numpy()
                    
        else:
            for iter, data in enumerate(data_loader):
                _, input, target, _, fnames, slices = data
                input = input.cuda(non_blocking=True)
                output = model(input)       
                output = output * cmask

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
    if inputs != None:
        for fname in inputs:
            inputs[fname] = np.stack(
                [out for _, out in sorted(inputs[fname].items())]
            )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])    
    num_subjects = len(reconstructions)
    
    if args.scheduler == 'Plateau':
        scheduler.step(metric_loss)  
    else:
        scheduler.step()
        
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, scaler, best_val_loss, is_new_best):
    try:
        torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / f'{args.exp_name+"_epoch"+str(epoch)}.pt'
    )
    except:
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
        shutil.copyfile(exp_dir / f'{args.exp_name+"_epoch"+str(epoch)}.pt', exp_dir / f'{args.exp_name}_best.pt')
        
    if os.path.exists(args.exp_dir / f'{args.exp_name+"_epoch"+str(epoch-1)}.pt'):
        os.remove(args.exp_dir / f'{args.exp_name+"_epoch"+str(epoch-1)}.pt')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)
            
        
def select_model(args):
    net_name = args.net_name.name
    if net_name == 'test_Unet':
        model = testUnet(in_chans = args.in_chans, out_chans = args.out_chans)
    elif net_name in ['newUnet', 'Unet']:
        model = Unet(in_chans = args.in_chans, out_chans = args.out_chans, alone=True)
    elif net_name == 'UnetPP':
        model = UnetPP(in_chans = args.in_chans, out_chans = args.out_chans)
    elif ('newUnet_' in net_name) or ('Unet_' in net_name):
        _, chans, pool_layer, drop = net_name.split('_')
        model = Unet(in_chans = args.in_chans, out_chans = args.out_chans,
                        chans = int(chans), num_pool_layers = int(pool_layer), drop_prob = float(drop))
    elif net_name == 'VarNet':
        assert args.input_key == 'kspace'
        model = VarNet(num_cascades=args.cascade)
        if args.load != '':
            return model
        
        VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
        MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
        if not Path(MODEL_FNAMES).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)

        pretrained = torch.load(MODEL_FNAMES)
        pretrained_copy = copy.deepcopy(pretrained)
        for layer in pretrained_copy.keys():
            if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
                del pretrained[layer]
                
        model.load_state_dict(pretrained)
        del pretrained, pretrained_copy
    elif net_name == 'AdaptiveVarNet':
        assert args.input_key == 'kspace'
        model = AdaptiveVarNet(num_cascades=args.cascade)
        
    else:
        raise Exception("Invalid Model was given as an argument :", net_name)
    
    return model


def freeze(layer):
    for params in layer.parameters():
        params.requires_grad = False

        
def freeze_layers(net, frozen_layers):
    for name, module in frozen_layers:
        module.apply(freeze)

        
def setting_transfer(net):
    modules = []
    for n, m in net.named_modules():
        if n != '' and n.count('.') == 1:
            modules.append((n,m))
    freeze_layers(net, modules[:-1])
    

def select_optimizer(args, model):
    if args.optim == 'Adam':
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=1e-5)
    elif args.optim == 'SGD':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, momentum=0.9, weight_decay=1e-5)
    elif args.optim == 'AdamW':
        return torch.optim.AdamW(params = filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    else:
        raise Exception("Invalid Optimizer was given as an argument :", args.optim)

        
def select_scheduler(args, optimizer):
    if args.scheduler in ['Plateau','P']:
        args.scheduler = 'Plateau'
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    elif args.scheduler in ['Cos','C']:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)
    else:
        raise Exception("Invalid Learning rate scheduler was given as an argument :", args.scheduler)
      
    
def train(args):    
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(' - Current .cuda device: ', torch.cuda.current_device())
    
    model = select_model(args)  
    model.to(device=device)
    if args.freeze:
        setting_transfer(model)
    
    loss_type = SSIMLoss().to(device=device) if args.loss == 'ssim' else nn.L1Loss().to(device=device)
    optimizer = select_optimizer(args, model)
    scheduler = select_scheduler(args, optimizer)
    scaler = GradScaler()
    start_epoch = 0 
    best_val_loss = 1. 
    
    if args.transfer != '' or args.load != '':
        load_path = args.transfer if args.transfer != '' else (args.exp_dir / args.load)
        print(f'\n*** Load Checkpoint for {load_path} loaded ***')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model'])
        try:
            scaler.load_state_dict(checkpoint['scaler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print("OPTIM LOAD FAILED")
        optimizer.param_groups[0]['lr'] = args.lr
        best_prev_loss = checkpoint['best_val_loss']
        if args.load:
            best_val_loss = best_prev_loss
        print(f"Start Epoch = {start_epoch+1}, prev best validation loss = {best_prev_loss:0.5f}")
        print(f"Previous learning rate was {checkpoint['optimizer']['param_groups'][0]['lr']}")
        print(f"Current learning rate is {optimizer.param_groups[0]['lr']}\n")    
    
    result = {}
    result['train_losses'] = []
    result['val_losses'] = []    
    
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, tv='train')
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, tv='val')

    
    for epoch in tqdm(range(start_epoch, start_epoch+args.num_epochs)):
        print(f'Epoch #{epoch+1:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scaler, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, scheduler)
        
        result['train_losses'].append(train_loss)
        result['val_losses'].append(val_loss/num_subjects)
        
#         train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scaler, best_val_loss, is_new_best)
        print('\n'
            f'* Epoch = [{epoch+1:4d}/{start_epoch+args.num_epochs:4d}] | Loss (Train/Val) = {train_loss:.4g} / {val_loss:.4g}'
            f"| Time(Train/Val) = {train_time:.4f}s / {val_time:.4f}s | Learning rate = {optimizer.param_groups[0]['lr']}",
        )
        
        if is_new_best:
            print("*** NewRecord ***")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f' - ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            
        
        save_exp_result(save_dir=args.json_dir, setting=deepcopy(vars(args)), result=deepcopy(result), load=args.load)
