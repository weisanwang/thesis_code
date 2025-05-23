import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Hippocampus segmentation training with continual learning')

    # net and criterion parameters
    parser.add_argument('--model', type=str, default='3DUNET', 
                        choices=['3DUNET', 'git_3DUNET'], help='Model for training')
    parser.add_argument('--criterion', type=str, default='GDL', 
                        choices=['DL', 'GDL'], help='Model for training')

    # basic parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_reduce_batch', type=int, default=3, help='learning rate reduce batches')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--batch_number', type=int, default=300, help='training batches each stage')
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers')
    parser.add_argument('--device', default='cuda:0', help='device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # LoRA parameters
    parser.add_argument('--lora', type=bool, default=False, help='use lora adapter or not')
    parser.add_argument('--lora_r', type=int, default=16, help='rank of LoRA')
    parser.add_argument('--lora_alpha', type=int, default=16, help='alpha of LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='dropout of LoRA')
    # Target modules for LoRA
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=["layer5.step.0", "layer5.step.3",
                                                                          "layer6.step.0", "layer6.step.3",
                                                                          "layer7.step.0", "layer7.step.3",
                                                                          "layer8"
                                                                          ], help='list of target modules for LoRA')

    # Shift detector parmeters
    parser.add_argument('--slide_window_length', type=int, default=7, help='sliding window length')
    parser.add_argument('--mean_threshold', type=float, default=0.25, help='mean_threshold')
    parser.add_argument('--var_threshold', type=float, default=1e-4, help='var_threshold')
    parser.add_argument('--jump_threshold', type=float, default=0.5, help='jump_threshold')
    
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--task_ids', type=int, nargs='+', default=[99, 98, 97], help='task ids')

    # Augmentation parameters
    parser.add_argument('--num_control_points', type=int, nargs=3, default=[4, 4, 4],help='num_control_points as "x,y,z"')
    parser.add_argument('--max_displacement', type=int, nargs=3, default=[2, 2, 1],help='maximal displacement as "x,y,z"')

    
    # continual learning parameters
    parser.add_argument('--cl_method', type=str, default='none', 
                        choices=['none', 'ewc', 'lwf'], help='Continual learning method')
    # EWC
    parser.add_argument('--ewc_lambda', type=float, default=1.0, help='ewc lambda')
    parser.add_argument('--ewc_alpha', type=float, default=0.9, help='ewc alpha')
    # LWF
    parser.add_argument('--lwf_lambda', type=float, default=1.0, help='lwf lambda')
    parser.add_argument('--lwf_temperature', type=int, default=5, help='lwf temperature')

    parser.add_argument('--task_number', type=int, default=3, help='task number')
    
    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    
    return parser
