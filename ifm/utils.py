import torch
import random
import numpy as np
from scipy.io import loadmat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = []
        self.top1 = []
        self.top5 = []
        self.time = []
        self.tp = []
        self.fp = []
        self.tn = []
        self.fn = []
        self.auc_roc = []

    def update(self, loss, top1, top5, time, tp=None, fp=None, tn=None, fn=None, auc_roc=None):
        self.loss.append(loss)
        self.top1.append(top1)
        self.top5.append(top5)
        self.time.append(time)
        if tp is not None:
            self.tp.append(tp)
        if fp is not None:
            self.fp.append(fp)
        if tn is not None:
            self.tn.append(tn)
        if fn is not None:
            self.fn.append(fn)
        if auc_roc is not None:
            self.auc_roc.append(auc_roc)

def recover_dict(data):
    """
    Recovers a dictionary loaded from a .mat file, converting arrays back to
    their original Python structures (lists, scalars, booleans, strings, etc.).
    """
    def recover_value(value):
        """Converts individual values to their appropriate Python type."""
        if isinstance(value, np.ndarray):
            # Handle scalar values
            if value.size == 1:
                item = value.item()
                return item if not (isinstance(item, float) and np.isnan(item)) else np.nan
            # Handle string arrays
            if value.dtype.type is np.str_ or value.dtype.type is np.object_:
                return value.tolist() if value.ndim > 0 else str(value.item())
            # Handle 2D row vectors
            if value.ndim == 2 and value.shape[0] == 1:
                return value.flatten().tolist()
            return value  # Return other arrays as is
        return value

    return {key: recover_value(value) for key, value in data.items() if not key.startswith('__')}

def get_acc(mat_fname):
    '''
    Get best accuracy
    '''
    res = recover_dict(loadmat(mat_fname))
    return np.max(res["top1"])