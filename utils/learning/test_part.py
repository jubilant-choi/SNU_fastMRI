import numpy as np
import torch

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders

from utils.model.varnet import VarNet
from utils.model.adaptive_varnet import AdaptiveVarNet
#from utils.model.unet_advanced import Unet
from utils.model.unet import Unet, UnetPP

def test(args, model, data_loader, boostNet=None):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)
    
    if boostNet != None:
        with torch.no_grad():
            with tqdm(data_loader, unit="batch", mininterval=2) as tepoch:
                tepoch.set_description(f"TEST ")
                for iter, data in enumerate(tepoch):

                    mask, kspace, _, _, fnames, slices = data
                    kspace = kspace.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)
                    output = boostNet(kspace, mask)
                    output = model(output)

                    for i in range(output.shape[0]):
                        reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
        return reconstructions, None
    
    elif args.net_name.name == 'VarNet':
        
        with torch.no_grad():
            with tqdm(data_loader, unit="batch", mininterval=2) as tepoch:
                tepoch.set_description(f"TEST ")
                for iter, data in enumerate(tepoch):

                    mask, kspace, _, _, fnames, slices = data
                    kspace = kspace.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)
                    output = model(kspace, mask)

                    for i in range(output.shape[0]):
                        reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
        return reconstructions, None

    elif args.net_name.name == 'Unet':
        with torch.no_grad():
            for (_, input, _, _, fnames, slices) in data_loader:
                input = input.cuda(non_blocking=True)
                output = model(input)

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                    inputs[fnames[i]][int(slices[i])] = input[i].cpu().numpy()

        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
        for fname in inputs:
            inputs[fname] = np.stack(
                [out for _, out in sorted(inputs[fname].items())]
            )
        return reconstructions, inputs


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())
    
    if args.net_name.name == 'VarNet':
        model = VarNet(num_cascades=args.cascade)
    elif args.net_name.name == 'AdaptiveVarNet':
        model = AdaptiveVarNet
    elif args.net_name.name == 'Unet':
        alone = True if args.boost else None
        model = Unet(in_chans = args.in_chans, out_chans = args.out_chans, alone = alone)
    elif net_name == 'UnetPP':
        model = UnetPP(in_chans = args.in_chans, out_chans = args.out_chans)
        
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / f'{args.exp_name}_best.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    boost = None
    
    if args.boost:
        print("*** boosting enabled ***")
        varnet_path = Path('./result/JB/VarNet/checkpoints/VarNet_test_sj_best.pt')
        boost = VarNet(num_cascades=2)
        boost.load_state_dict(torch.load(varnet_path,map_location='cpu')['model'])
        boost.to(device=device)
        
        args.input_key = 'kspace'
        args.net_name = Path('VarNet')
        
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, tv='test',isforward = True)
    reconstructions, inputs = test(args, model, forward_loader, boost)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)