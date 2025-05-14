import torch
import torchio as tio
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter
from CL_task_agnostic.CL_function import EWCpp
from CL_task_agnostic.shift_detector import ShiftDetector

def evaluate_batch(model: torch.nn.Module, val_criterion, data_loader: Iterable, device: torch.device, writer, writer_batch: int, args):
    model.eval()
    for i,task_id in enumerate(args.task_ids):
    # for task_id in range(args.num_tasks):
        test_loader = data_loader[i]['val']
        # Calculate the sum dice score of each test dataset
        dice_score_sum = 0.0
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                images = data['mri'][tio.DATA].to(device)
                labels = data['segmentation'][tio.DATA].to(device)
                logits = model(images)
                dice_score_sum += val_criterion(logits,labels)
            average_dice_score = dice_score_sum / len(test_loader)
            print(f"Batch {writer_batch} on Task {task_id} got Dice Score: {average_dice_score}")
            writer.add_scalar(f'Testing{task_id}', average_dice_score, writer_batch)


def train_and_evaluate(model: torch.nn.Module, criterion, val_criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler, device: torch.device, args):

    count_updates=0
    batch=[]
    # Loss detection
    loss_window=[]
    loss_window_means=[]
    loss_window_variances=[]
    new_peak_detected=True
    detector = ShiftDetector(
        slide_window_length=args.slide_window_length, 
        mean_threshold=args.mean_threshold, 
        var_threshold=args.var_threshold)

    # Define path of results output dir
    exp_name = "_".join([f"{key}={value}" for key, value in vars(args).items() if key in [
        "lr","lr_reduce_batch", "batch_size", "batch_number", "seed", "num_control_points", "max_displacement"
        ]])
    if args.lora == False:
        logdir = f"./log_results/log_CL_task_agnostic/{args.cl_method}/{exp_name}"
    else:
        logdir = f"./log_results/log_CL_task_agnostic/{args.cl_method}_lora/{exp_name}"
    
    writer = SummaryWriter(log_dir=logdir)


    # Training
    writer_batch = 0
    for i, _ in enumerate(args.task_ids):
        print(f"Training start =========================================")
        for _, data in enumerate(data_loader[i]['train']):
            writer_batch += 1
            # Training
            model.train()
            images = data['mri'][tio.DATA].to(device)
            labels = data['segmentation'][tio.DATA].to(device)
            logits = model(images)

            # Basic loss
            loss_batch = criterion(logits, labels)
            loss_for_peak_detector = loss_batch.item() 

            # Detect if domain shift
            if detector.update(loss_for_peak_detector):
                # apply ewc
                


            # Calculate the total loss of current batch
            if args.cl_method == 'None':
                total_loss = loss_batch
            elif args.cl_method == 'ewc':
                ewc_loss = args.ewc_lambda * ewc.penalty()  
                writer.add_scalar(f'ewc_loss', ewc_loss, writer_batch)
                total_loss = criterion(logits, labels) + ewc_loss
            # elif args.cl_method == 'lwf':
            #     lwf_loss = args.lwf_lambda * lwf.penalty()
            #     writer.add_scalar(f'lwf_loss', lwf_loss, writer_batch)
            #     total_loss = criterion(logits, labels) + lwf_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Batch {writer_batch} get loss: {total_loss}")
            # print(f"Batch {writer_batch} get learning rate: {current_lr}")
            writer.add_scalar(f'learning rate', current_lr, writer_batch)
            writer.add_scalar(f'train_loss', total_loss, writer_batch)


            # Testing
            evaluate_batch(model, val_criterion, data_loader, device, writer, writer_batch, args)

            if writer_batch % args.lr_reduce_batch == 0:
                scheduler.step()



