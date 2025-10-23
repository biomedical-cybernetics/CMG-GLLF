import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import Metrics, setup_seed
from Train_and_Test import Train, Test
from data_loader import load_data_mlp
import os
from scipy.io import savemat
import time
import argparse
from tqdm import tqdm
from models_tuner import *


def create_scheduler(optimizer, scheduler_type, epochs):
    """
    Create a learning rate scheduler based on the scheduler type.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ('linear', 'step', 'exponential', 'cosine', 'none')
        epochs: Total number of training epochs
    
    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type == 'linear':
        return torch.optim.lr_scheduler.LinearLR(optimizer,
                                               start_factor=1.0,
                                               end_factor=0.01,
                                               total_iters=int(epochs * 0.9))
    elif scheduler_type == 'exponential_95':
        stop_epoch = int(epochs * 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                               lambda epoch: 0.95 ** min(epoch, stop_epoch))
    elif scheduler_type == 'exponential_97':
        stop_epoch = int(epochs * 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                               lambda epoch: 0.97 ** min(epoch, stop_epoch))
    elif scheduler_type == 'exponential_99':
        stop_epoch = int(epochs * 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                               lambda epoch: 0.99 ** min(epoch, stop_epoch))    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=epochs // 3,
                                             gamma=0.1)
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                    gamma=0.95)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=epochs,
                                                        eta_min=0.001)
    elif scheduler_type == 'exponential_to_linear':
        # Exponential decay that reaches 0.01 at 90% of epochs (same as linear)
        stop_epoch = int(epochs * 0.9)
        # Calculate gamma such that gamma^stop_epoch = 0.01
        gamma = 0.01 ** (1.0 / stop_epoch)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: gamma ** min(epoch, stop_epoch)
        )
    elif scheduler_type == 'power_to_linear':
        stop_epoch = int(epochs * 0.9)
        power = 0.5  # You can adjust this power as needed
        def power_decay(epoch):
            if epoch >= stop_epoch:
                return 0.01
            t = epoch / stop_epoch
            return 0.01 + (1.0 - 0.01) * ((1 - t) ** power)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, power_decay)    
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_fname(ds,network,bs,lr_w,lr_special,epochs,seed,use_tuner,mode,train_2,init_method,weight_decay,rectify,single_param,no_special_decay=False,scheduler_w='linear',scheduler_special='none'):
    '''
    get file names
    '''
    if use_tuner:
        tuner_signature=mode
        # For modes without learnable parameters, don't include lr_special in filename
        if mode in ['sigmoid', 'tanh']:
            lr_part = f"lr_w_{lr_w}"
        else:
            lr_part = f"lr_w_{lr_w}_lr_special_{lr_special}"
        
        elements = os.path.join(
            ds,
            network,
            f"e_{epochs}",
            f"bs_{bs}",
            lr_part,
            f"seed_{seed}",
            f"train_2_{train_2}",
            f"init_{init_method}",
            f"rectify_{rectify}",
            f"single_{single_param}",
            tuner_signature
        )
    else:
        elements = os.path.join(
            ds,
            network,
            f"e_{epochs}",
            f"bs_{bs}",
            f"lr_w_{lr_w}",
            f"seed_{seed}",
            "no_tuner"
        )
    # NOTE: if you want to use weight decay, please add it to the elements
    if weight_decay > 0:
        elements = os.path.join(elements, f"wd_{weight_decay}")
    if no_special_decay:
        elements = os.path.join(elements, "no_special_decay")
    # Add scheduler information only if non-default schedulers are used
    if scheduler_w != 'linear' or scheduler_special != 'none':
        if scheduler_w != 'linear':
            elements = os.path.join(elements, f"sched_w_{scheduler_w}")
        if scheduler_special != 'none':
            elements = os.path.join(elements, f"sched_special_{scheduler_special}")
    path = os.path.join(
        './results',
        'tuner',
        elements
    )
    log_fname = os.path.join('./logs', 'tuner', elements)
    return path, log_fname

def train_model(ds,network,bs,lr_w,lr_special,epochs,seed,use_tuner,mode,train_2,init_method,device,weight_decay,rectify,single_param,no_special_decay,scheduler_w,scheduler_special):
    setup_seed(seed)
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    save_path, log_fname = get_fname(ds,network,bs,lr_w,lr_special,epochs,seed,use_tuner,mode,train_2,init_method,weight_decay,rectify,single_param,no_special_decay,scheduler_w,scheduler_special)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(log_fname), exist_ok=True)
    save_fname_test = os.path.join(save_path, 'res_test.mat')
    save_fname_train = os.path.join(save_path, 'res_train.mat')
    if os.path.exists(save_fname_train):
        print(f"Training results already exist at {save_fname_train}. Skipping training.")
        return

    writer = SummaryWriter(log_dir=log_fname)
    
    train_loader, test_loader, indim, outdim, _ = load_data_mlp(ds,bs,dim_factor=2)
    model_dict = {
        'MLP_CIFAR10_A': MLP_CIFAR10_A,
        'MLP_CIFAR100_A': MLP_CIFAR100_A,
        'MLP_CIFAR10_B': MLP_CIFAR10_B,
        'MLP_CIFAR100_B': MLP_CIFAR100_B,
        'LeNet5_CIFAR': LeNet5_CIFAR,
        'MLP_TIN': MLP_TIN
    }
    model_class = model_dict.get(network)
    if model_class is None:
        raise ValueError(f"Unknown network architecture: {network}")
    
    model = model_class(indim, outdim, use_tuner, mode, train_2, init_method, ds, single_param).to(device)
    # 分离参数
    main_params = []
    special_params = []

    for param in model.parameters():
        # 检查自定义标识
        if hasattr(param, 'is_special_param'):
            special_params.append(param)
        else:
            main_params.append(param)    
    #设置优化器
    if use_tuner and lr_special is not None:
        # If tuner is enabled, use lr_special for special params
        if no_special_decay:
            # Create separate optimizers for main and special parameters
            main_optimizer = torch.optim.SGD(main_params, lr=lr_w, momentum=0.9, weight_decay=weight_decay)
            special_optimizer = torch.optim.SGD(special_params, lr=lr_special, momentum=0.9, weight_decay=0)
            optimizer = [main_optimizer, special_optimizer]
        else:
            # Use single optimizer with parameter groups
            optimizer = torch.optim.SGD([
                    {'params': main_params, 'lr': lr_w, 'weight_decay':weight_decay},     # 常规权重
                    {'params': special_params, 'lr': lr_special, 'weight_decay': 0}   # mu_raw/I_raw params
                ], momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_w, momentum=0.9, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    if no_special_decay and use_tuner and lr_special is not None:
        # Create scheduler for main parameters only
        scheduler = create_scheduler(optimizer[0], scheduler_w, epochs)  # main_optimizer
        special_scheduler = create_scheduler(optimizer[1], scheduler_special, epochs)  # special_optimizer
    else:
        # Create scheduler for all parameters
        scheduler = create_scheduler(optimizer, scheduler_w, epochs)
        special_scheduler = None
    
    # start training
    m_test = Metrics()
    m_train = Metrics()
    best_acc=0.0
    for epoch in range(epochs):
        time_start = time.time()
        
        # Handle different optimizer structures
        if no_special_decay and use_tuner and lr_special is not None:
            # Use custom training loop for multiple optimizers
            model.train()
            CSE = nn.CrossEntropyLoss().to(device)
            for batch_idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
                if len(batch_data) == 3:
                    data, target, _ = batch_data  # Unpack 3 values and ignore filename
                else:
                    data, target = batch_data  # Unpack 2 values for other datasets        
                data, target = data.to(device), target.to(device)
                
                # Zero gradients for both optimizers
                optimizer[0].zero_grad()  # main_optimizer
                optimizer[1].zero_grad()  # special_optimizer
                
                output = model(data)
                loss = CSE(output, target)
                loss.backward()
                
                # Step both optimizers
                optimizer[0].step()  # main_optimizer
                optimizer[1].step()  # special_optimizer
        else:
            # Use standard Train function for single optimizer
            Train(model, device, train_loader, optimizer, None)
            
        time_consum = time.time() - time_start
        # On test set
        top1_test, top5_test, loss_test, tp_test, fp_test, tn_test, fn_test, auc_roc_test = Test(model, device, test_loader)
        m_test.update(loss_test, top1_test, top5_test, time_consum, tp_test, fp_test, tn_test, fn_test, auc_roc_test)     
        # On train set
        top1_train, top5_train, loss_train, tp_train, fp_train, tn_train, fn_train, auc_roc_train = Test(model, device, train_loader)
        m_train.update(loss_train, top1_train, top5_train, time_consum, tp_train, fp_train, tn_train, fn_train, auc_roc_train)        
        print(f"Epoch {epoch} consume {time_consum}s, ACC Test={top1_test} ACC Train={top1_train}")

        if top1_test > best_acc:
            best_acc = top1_test  
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        
        # Step schedulers
        if scheduler is not None:
            scheduler.step()
        if special_scheduler is not None:
            special_scheduler.step()
            
        writer.add_scalar('Top1/train',top1_train, epoch)
        writer.add_scalar('loss/train',loss_train, epoch)    
        writer.add_scalar('Top1/test',top1_test, epoch)
        writer.add_scalar('loss/test',loss_test, epoch)    
    # record also the fp, tp, tn, fn, and auc_roc
    savemat(save_fname_test, {'top1':m_test.top1, 'top5':m_test.top5, 'loss':m_test.loss, 'time':m_test.time,
                            'best_acc':best_acc,'aae':float(np.trapz(m_test.top1)/(len(m_test.top1)-1)),
                            'tp':m_test.tp, 'fp':m_test.fp, 'tn':m_test.tn, 'fn':m_test.fn,
                            'auc_roc':m_test.auc_roc,
                            'num_params':sum(p.numel() for p in model.parameters() if p.requires_grad)})   
    # save the data for train set
    savemat(save_fname_train, {'top1':m_train.top1, 'top5':m_train.top5, 'loss':m_train.loss, 'time':m_train.time,
                            'best_acc':np.max(m_train.top1),'aae':float(np.trapz(m_train.top1)/(len(m_train.top1)-1)),
                            'tp':m_train.tp, 'fp':m_train.fp, 'tn':m_train.tn, 'fn':m_train.fn,
                            'auc_roc':m_train.auc_roc,
                            'num_params':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    
    # Close TensorBoard writer
    writer.close()  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP with different learning rates for weights and tuner parameters')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use (MNIST, EMNIST, CIFAR10)')
    parser.add_argument('--network', type=str, default='MLP_CIFAR10_A', help='Network architecture to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr_w', type=float, default=0.01, help='Learning rate for weights')
    parser.add_argument('--lr_special', type=float, help='Learning rate for tuner parameters')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--use_tuner', action='store_true', help='Use tuner for input transformation')
    parser.add_argument('--mode', type=str, default='CM-GLLF-logistic', help='Tuner mode (linear, cubic, CM-GLLF-logistic)')
    parser.add_argument('--train_2', action='store_true', help='Train second parameter (b for linear/cubic, I for CM-GLLF)')
    parser.add_argument('--init_method', type=str, default='define', help='Initialization method (define, uniform)')
    parser.add_argument('--device',type=int, default=0, help='Device to use (GPU index)')
    parser.add_argument('--weight_decay',type=float, default=0, help='weight decay for optimizer')
    parser.add_argument('--rectify', action='store_true', help='Apply ReLU after tuner')
    parser.add_argument('--single_param', action='store_true', help='Use single parameter for all pixels')
    parser.add_argument('--no_special_decay', action='store_true', help='Prevent special parameters learning rate from decaying')
    parser.add_argument('--scheduler_w', type=str, default='linear', help='Learning rate scheduler for weights (linear, step, exponential, cosine, none)')
    parser.add_argument('--scheduler_special', type=str, default='none', help='Learning rate scheduler for special parameters (linear, step, exponential, cosine, none)')
    args = parser.parse_args()
    
    train_model(args.dataset, args.network, args.batch_size, args.lr_w, args.lr_special, 
               args.epochs, args.seed, args.use_tuner, args.mode, args.train_2, 
               args.init_method, args.device, args.weight_decay, args.rectify, args.single_param, args.no_special_decay, args.scheduler_w, args.scheduler_special)
