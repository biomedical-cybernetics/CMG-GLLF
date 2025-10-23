import torch.nn as nn
import torch
import torch.distributed as dist
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def Train(model, device, train_loader, optimizer,mode):
    '''
    Train for neuronwise CM-GLLF
    '''
    model.train()
    CSE = nn.CrossEntropyLoss().to(device)
    for batch_idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        if len(batch_data) == 3:
            data, target, _ = batch_data  # Unpack 3 values and ignore filename
        else:
            data, target = batch_data  # Unpack 2 values for other datasets        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CSE(output, target)
        loss.backward()
        # if mode is 'logit', clamp the gradient of mu_raw to avoid too large update
        if mode == 'logit':
            for module in model.modules():
                if hasattr(module, 'mu_raw') and module.mu_raw.grad is not None:
                    module.mu_raw.grad.data.clamp_(-1.0, 1.0)
        optimizer.step()


def Test(model, device, val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Initialize lists to collect all predictions and targets for confusion matrix and AUC-ROC
    all_predictions = []
    all_targets = []
    all_probabilities = []

    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    end = time.time()
    for i, batch_data in enumerate(val_loader):
        if len(batch_data) == 3:
            input, target, _ = batch_data  # Unpack 3 values and ignore filename
        else:
            input, target = batch_data  # Unpack 2 values for other datasets
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # determine the number of classes
            n_classes = output.size(1)
            topk = (1, min(5, n_classes))  # Use min(5, n_classes) to avoid error
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=topk)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            # Collect predictions and targets for confusion matrix and AUC-ROC
            _, predicted = torch.max(output.data, 1)
            probabilities = torch.softmax(output, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # Compute confusion matrix metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Get unique classes
    unique_classes = np.unique(all_targets)
    n_classes = len(unique_classes)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=unique_classes)
    
    # Initialize metrics
    tp_total = fp_total = tn_total = fn_total = 0
    auc_scores = []
    
    # For multi-class, compute metrics for each class and then average
    for i, class_label in enumerate(unique_classes):
        # Binary classification for each class (one-vs-rest)
        binary_targets = (all_targets == class_label).astype(int)
        binary_predictions = (all_predictions == class_label).astype(int)
        
        # Compute TP, FP, TN, FN for this class
        tp = np.sum((binary_targets == 1) & (binary_predictions == 1))
        fp = np.sum((binary_targets == 0) & (binary_predictions == 1))
        tn = np.sum((binary_targets == 0) & (binary_predictions == 0))
        fn = np.sum((binary_targets == 1) & (binary_predictions == 0))
        
        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn
        
        # Compute AUC-ROC for this class (one-vs-rest)
        if n_classes > 2:
            try:
                auc = roc_auc_score(binary_targets, all_probabilities[:, i])
                auc_scores.append(auc)
            except ValueError:
                # Handle case where class doesn't appear in targets
                pass
        
    # For binary classification, compute AUC-ROC directly
    if n_classes == 2:
        try:
            auc_roc = roc_auc_score(all_targets, all_probabilities[:, 1])
        except ValueError:
            auc_roc = 0.5  # Default value if AUC cannot be computed
    else:
        # For multi-class, use macro-average AUC
        auc_roc = np.mean(auc_scores) if auc_scores else 0.5

    top1 = top1.avg.cpu().numpy()
    top5 = top5.avg.cpu().numpy()
    loss = losses.avg
    
    return top1, top5, loss, tp_total, fp_total, tn_total, fn_total, auc_roc