import os
import torch
import torchio as tio
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter

from CL_task_agnostic.shift_detector import ShiftDetector
from CL_task_agnostic.CL_function import CLStrategy, EWCPlusStrategy, LwFStrategy

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
    # Loss detector
    shift_detector = ShiftDetector(
        slide_window_length=args.slide_window_length, 
        mean_threshold=args.mean_threshold, 
        var_threshold=args.var_threshold
        )

    # Define path of results output dir
    exp_name = "_".join([f"{key}={value}" for key, value in vars(args).items() if key in [
        "lr","lr_reduce_batch", "batch_size", "batch_number", "seed", "num_control_points", "max_displacement"
        ]])
    if args.lora == False:
        logdir = f"./log_results/log_CL_task_agnostic/{args.cl_method}/{exp_name}"
    else:
        logdir = f"./log_results/log_CL_task_agnostic/{args.cl_method}_lora/{exp_name}"
    
    # Create tensorboard writer to save the results
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    # Create a txt document to save the peak detection results
    txt_log = os.path.join(logdir, 'domain_shifts.txt')

    if args.cl_method == 'ewc':
        cl_strategy = EWCPlusStrategy(
            device=device,
            alpha=args.ewc_alpha,
            ewc_lambda=args.ewc_lambda,
            normalize=True
        )
    elif args.cl_method == 'lwf':  
        pass
    elif args.cl_method == 'none':
        cl_strategy = CLStrategy()

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

            # Detect if domain shift
            loss_for_peak_detector = loss_batch.item() 
            if shift_detector.update(loss_for_peak_detector):
                cl_strategy.on_domain_shift(model)
                # Write the detected peak into the txt file
                with open(txt_log, 'a') as f:
                    f.write(f"Batch {writer_batch} got the {shift_detector.peak_count} peak\n")
                
            optimizer.zero_grad()
            # Get the gradient of the basic loss
            loss_batch.backward()

            # Update the CL strategy 
            cl_strategy.after_backward(model)

            # CL strategy loss 
            loss_cl = cl_strategy.penalty(model, student_logits=logits, inputs=images)
            if loss_cl.item() != 0.0:
                loss_cl.backward()

            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Batch {writer_batch} get basic loss: {loss_batch.item()} and CL loss: {loss_cl.item()}")
            # print(f"Batch {writer_batch} get learning rate: {current_lr}")
            writer.add_scalar(f'learning rate', current_lr, writer_batch)
            writer.add_scalar(f'total train loss', loss_batch.item() + loss_cl.item(), writer_batch)
            writer.add_scalar(f'CL loss', loss_cl.item(), writer_batch)


            # Testing
            evaluate_batch(model, val_criterion, data_loader, device, writer, writer_batch, args)

            if writer_batch % args.lr_reduce_batch == 0:
                scheduler.step()



