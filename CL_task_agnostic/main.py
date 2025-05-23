import os
import torch
import torch.backends.cudnn as cudnn
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

    # cudnn.benchmark = True

    splited_dataset = split_hippocampus_datasets(args)

    data_loader = build_continual_dataloader(splited_dataset, args)
    # print(f"Data loaded with {len(data_loader)} tasks")

    # Creat training model
    net = UNet3D().to(device)
    print(args)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
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