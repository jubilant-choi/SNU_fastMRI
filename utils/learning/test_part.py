import numpy as np
import torch

from collections import defaultdict
from tqdm import tqdm

from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders

from utils.model.varnet import VarNet
from utils.model.adaptive_varnet import AdaptiveVarNet
#from utils.model.unet_advanced import Unet
from utils.model.unet import Unet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    inputs = defaultdict(dict)
    
    if args.net_name.name == 'VarNet':
        
        with torch.no_grad():
            with tqdm(data_loader, unit="batch") as tepoch:
                for iter, data in enumerate(tepoch):
                    tepoch.set_description(f"TEST ")

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

    elif args.net_name.name == 'UNet':
        with torch.no_grad():
            for (input, _, _, fnames, slices) in data_loader:
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
        model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
        
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / f'{args.exp_name}_best.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)