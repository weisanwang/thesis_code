import numpy as np
import os
import torch
from CL_task_agnostic.datasets import build_continual_dataloader, set_seed, split_hippocampus_datasets
from CL_task_agnostic.engine import train_and_evaluate
from CL_task_agnostic.hippocampus_config import get_args_parser
from model.model import UNet3D, dice_loss_logit, dice_coefficient_logit

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

def main(args):

    device = torch.device(args.device)

    seed = args.seed
    set_seed(seed)

    # Prepare the training and testing dataset 
    splited_dataset = split_hippocampus_datasets(args)
    # Build the dataloader
    data_loader = build_continual_dataloader(splited_dataset, args)

    # Creat training model
    net = UNet3D().to(device)
    print('\n Current used hyper-parameters are shown as below:', args)

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # Define the loss function and validation metric
    criterion = dice_loss_logit()
    val_criterion = dice_coefficient_logit()

    train_and_evaluate(
        model=net, 
        criterion=criterion,
        val_criterion=val_criterion, 
        data_loader=data_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=device, 
        args=args
        )


if __name__ == '__main__':

    args = get_args_parser().parse_args()
    main(args)