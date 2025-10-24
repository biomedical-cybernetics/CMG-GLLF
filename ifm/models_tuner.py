from tuners import (Linear_tuner, Cubic_tuner, CM_GLLF_tuner, Linear_tuner_valuewise, Cubic_tuner_valuewise, CM_GLLF_tuner_valuewise,
                     CM_GLLF_tuner_clamp_valuewise, AdaptiveClassic_tuner_valuewise, SReLU, CM_GLLF_tuner_valuewise_restrict,
                     CM_GLLF_tuner_flexible, CM_GLLF_tuner_trainable_minmax) #, CM_GLLF_tuner_all_valuewise
import torch.nn as nn


def get_tuner(mode, indim, train_2, init_method, dataset, rectify=False, single_param=False):
    '''
    train_2 indicates whether train the second parameter
    '''

    if mode == 'adaptive-sigmoid-valuewise':
        return AdaptiveClassic_tuner_valuewise(indim, train_2, init_method, dataset, mode='Sigmoid', rectify=rectify, single_param=single_param)
    elif mode == 'adaptive-tanh-valuewise':
        return AdaptiveClassic_tuner_valuewise(indim, train_2, init_method, dataset, mode='tanh', rectify=rectify, single_param=single_param)
    elif mode == 'srelu_valuewise_positive':
        return SReLU(rectify=False, neuronwise=not single_param, num_neurons=indim if not single_param else 1, init_method=init_method, positive_slopes=True)        
    elif mode == 'linear-valuewise-positive':
        return Linear_tuner_valuewise(indim, train_2, init_method, dataset, rectify, single_param, positive_a=True)
    elif mode == 'cubic-valuewise-positive':
        return Cubic_tuner_valuewise(indim, train_2, init_method, dataset, rectify, single_param, positive_a=True)
    elif mode == 'CM-GLLF-all-valuewise-minmax-mapping':
        return CM_GLLF_tuner_flexible(indim, train_2, init_method, dataset, minmax_setting='minmax_mapping', mode='all' ,rectify=rectify, offset=1e-6)              
    else:
        raise ValueError(f"Invalid tuner mode: {mode}")

class MLP_CIFAR10_A(nn.Module):
    '''
    MLP for CIFAR10
    MLP with 2 hidden layers
    Hidden layer sizes are 1024 and 512
    Uses dropout with p=0.3
    from Nerva: a Truly Sparse Implementation of Neural Networks
    https://arxiv.org/abs/2407.17437
    '''
    def __init__(self, indim, outdim, use_tuner, mode, train_2, init_method, dataset, single_param) -> None:
        super().__init__()
        self.use_tuner = use_tuner
        if use_tuner:
            self.tuner = get_tuner(mode, indim, train_2, init_method, dataset, single_param=single_param)
        hiddim = [1024, 512]
        self.Linear1 = nn.Linear(indim, hiddim[0])
        self.Linear2 = nn.Linear(hiddim[0], hiddim[1])
        self.last_layer = nn.Linear(hiddim[1], outdim)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_tuner:
            out = self.tuner(x.reshape(batch_size, -1))
        else:
            out = x.reshape(batch_size, -1)
        out = self.act1(self.Linear1(out))
        out = self.dropout1(out)
        out = self.act2(self.Linear2(out))
        out = self.dropout2(out)
        out = self.last_layer(out)
        return out


class MLP_CIFAR10_B(nn.Module):
    '''
    MLP for CIFAR10
    MLP with 2 hidden layers
    Hidden layer sizes are 1024 and 512
    Uses dropout with p=0.3
    ReLU is without inplace operation
    from Nerva: a Truly Sparse Implementation of Neural Networks
    https://arxiv.org/abs/2407.17437
    '''
    def __init__(self, indim, outdim, use_tuner, mode, train_2, init_method, dataset, single_param) -> None:
        super().__init__()
        self.use_tuner = use_tuner
        if use_tuner:
            self.tuner = get_tuner(mode, indim, train_2, init_method, dataset, single_param=single_param)
        hiddim = [1024, 512]
        self.Linear1 = nn.Linear(indim, hiddim[0])
        self.Linear2 = nn.Linear(hiddim[0], hiddim[1])
        self.last_layer = nn.Linear(hiddim[1], outdim)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_tuner:
            out = self.tuner(x.reshape(batch_size, -1))
        else:
            out = x.reshape(batch_size, -1)
        out = self.act1(self.Linear1(out))
        out = self.dropout1(out)
        out = self.act2(self.Linear2(out))
        out = self.dropout2(out)
        out = self.last_layer(out)
        return out


class MLP_CIFAR100_A(nn.Module):
    '''
    MLP for CIFAR100
    MLP with 2 hidden layers
    Hidden layer sizes are 2048 and 1024
    Uses dropout with p=0.3
    '''
    def __init__(self, indim, outdim, use_tuner, mode, train_2, init_method, dataset, single_param) -> None:
        super().__init__()
        self.use_tuner = use_tuner
        if use_tuner:
            self.tuner = get_tuner(mode, indim, train_2, init_method, dataset, single_param=single_param)
        hiddim = [2048, 1024]
        self.Linear1 = nn.Linear(indim, hiddim[0])
        self.Linear2 = nn.Linear(hiddim[0], hiddim[1])
        self.last_layer = nn.Linear(hiddim[1], outdim)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_tuner:
            out = self.tuner(x.reshape(batch_size, -1))
        else:
            out = x.reshape(batch_size, -1)
        out = self.act1(self.Linear1(out))
        out = self.dropout1(out)
        out = self.act2(self.Linear2(out))
        out = self.dropout2(out)
        out = self.last_layer(out)
        return out
    

class MLP_CIFAR100_B(nn.Module):
    '''
    MLP for CIFAR100
    MLP with 2 hidden layers
    Hidden layer sizes are 2048 and 1024
    Uses dropout with p=0.3
    ReLU is without inplace operation
    '''
    def __init__(self, indim, outdim, use_tuner, mode, train_2, init_method, dataset, single_param) -> None:
        super().__init__()
        self.use_tuner = use_tuner
        if use_tuner:
            self.tuner = get_tuner(mode, indim, train_2, init_method, dataset, single_param=single_param)
        hiddim = [2048, 1024]
        self.Linear1 = nn.Linear(indim, hiddim[0])
        self.Linear2 = nn.Linear(hiddim[0], hiddim[1])
        self.last_layer = nn.Linear(hiddim[1], outdim)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_tuner:
            out = self.tuner(x.reshape(batch_size, -1))
        else:
            out = x.reshape(batch_size, -1)
        out = self.act1(self.Linear1(out))
        out = self.dropout1(out)
        out = self.act2(self.Linear2(out))
        out = self.dropout2(out)
        out = self.last_layer(out)
        return out
    

class LeNet5_CIFAR(nn.Module):
    '''
    LeNet5 structure designed for CIFAR-10 and CIFAR-100
    '''
    def __init__(self, indim, outdim, use_tuner, mode, train_2, init_method, dataset, single_param) -> None:
        super().__init__()
        self.use_tuner = use_tuner
        if use_tuner:
            self.tuner = get_tuner(mode, indim, train_2, init_method, dataset, single_param=single_param)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # input channels=3 for RGB images
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted for CIFAR image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, outdim)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_tuner:
            # first flatten x, use tuner, and then reshape back
            batch_size = x.shape[0]
            original_shape = x.shape
            x_flat = x.reshape(batch_size, -1)
            x_tuned = self.tuner(x_flat)
            x = x_tuned.reshape(original_shape)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x
    

class MLP_TIN(nn.Module):  
    '''
    MLP for Tiny ImageNet
    MLP with 3 hidden layers
    Hidden layer sizes are 4096, 2048, and 1024
    Uses dropout with p=0.3
    '''
    def __init__(self, indim, outdim, use_tuner, mode, train_2, init_method, dataset, single_param) -> None:
        super().__init__()
        self.use_tuner = use_tuner
        if use_tuner:
            self.tuner = get_tuner(mode, indim, train_2, init_method, dataset, single_param=single_param)        
        hiddim = [4096, 2048, 1024]
        self.Linear1 = nn.Linear(indim, hiddim[0])
        self.Linear2 = nn.Linear(hiddim[0], hiddim[1])
        self.Linear3 = nn.Linear(hiddim[1], hiddim[2])
        self.last_layer = nn.Linear(hiddim[2], outdim)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_tuner:
            out = self.tuner(x.reshape(batch_size, -1))
        else:
            out = x.reshape(batch_size, -1)
        out = self.act1(self.Linear1(out))
        out = self.dropout1(out)
        out = self.act2(self.Linear2(out))
        out = self.dropout2(out)
        out = self.act3(self.Linear3(out))
        out = self.dropout3(out)
        out = self.last_layer(out)
        return out