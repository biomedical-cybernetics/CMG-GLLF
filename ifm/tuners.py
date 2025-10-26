#%% giving each pixel(containing 3 RGB values) a unique function to tune
import torch
import torch.nn as nn
from torch.autograd import Function


#%% giving each RGB value a unique function to tune
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_tuner_valuewise(nn.Module):
    '''
    Linear activation function: ax + b
    For define method: a=1, b=0
    For uniform method: a is uniformly sampled from (0,1), b is uniformly sampled from (-1,1)
    For normal method: a ~ N(1, 0.1^2), b ~ N(0, 0.1^2)
    Here for each RGB value we give a unique CM-GLLF function
    '''
    def __init__(self, indim, train_b, init_method, dataset, rectify=False, single_param=False, positive_a=False):
        super().__init__()
        assert init_method in ['define','uniform','normal']
        
        # For colored images, we want one linear function per pixel (shared across color channels)
        if single_param:
            n_params = 1  # Single parameter for all pixels
        else:
            n_params = indim
            
        if init_method == 'define':
            init_a = torch.ones((n_params,))
            init_b = torch.zeros((n_params,))
        elif init_method == 'uniform':
            init_a = torch.empty((n_params,)).uniform_(0, 1)
            init_b = torch.empty((n_params,)).uniform_(-1, 1)
        elif init_method == 'normal':
            init_a = torch.normal(1.0, 0.1, (n_params,))
            init_b = torch.normal(0.0, 0.1, (n_params,))
            
        self.rectify = rectify
        self.train_b = train_b
        self.dataset = dataset
        self.single_param = single_param
        self.positive_a = positive_a
        
        # Initialize learnable parameters
        if positive_a:
            # Use log for a to ensure it stays positive
            self.a_raw = nn.Parameter(torch.log(torch.clamp(init_a, min=1e-6)))
            self.a_raw.is_special_param = True
        else:
            self.a = nn.Parameter(init_a)
            self.a.is_special_param = True
        
        if self.train_b:
            self.b = nn.Parameter(init_b)
            self.b.is_special_param = True
        else:
            self.register_buffer('b', init_b)
                
        self.indim = indim

    def forward(self, x):
        # For colored datasets, expand parameters to match input dimensions
        if self.single_param:
            # Use the same parameters for all pixels
            if self.positive_a:
                a_expanded = self.a_raw.exp().expand(self.indim)
            else:
                a_expanded = self.a.expand(self.indim)
            if self.train_b:
                b_expanded = self.b.expand(self.indim)
            else:
                b_expanded = self.b.expand(self.indim)
        else:
            if self.positive_a:
                a_expanded = self.a_raw.exp()
            else:
                a_expanded = self.a
            b_expanded = self.b
            
        # Simple linear transformation: ax + b
        y = a_expanded * x + b_expanded
        return torch.clamp(y, min=0) if self.rectify else y
    

class Cubic_tuner_valuewise(nn.Module):
    '''
    Cubic activation function: ax^3 + b
    For define method: a=1, b=0
    For uniform method: a is uniformly sampled from (0,1), b is uniformly sampled from (-1,1)
    For normal method: a ~ N(1, 0.1^2), b ~ N(0, 0.1^2)
    '''
    def __init__(self, indim, train_b, init_method, dataset, rectify=False, single_param=False, positive_a=False):
        super().__init__()
        assert init_method in ['define','uniform','normal']
        
        # For colored images, we want one linear function per pixel (shared across color channels)
        if single_param:
            n_params = 1  # Single parameter for all pixels
        else:
            n_params = indim
            
        if init_method == 'define':
            init_a = torch.ones((n_params,))
            init_b = torch.zeros((n_params,))
        elif init_method == 'uniform':
            init_a = torch.empty((n_params,)).uniform_(0, 1)
            init_b = torch.empty((n_params,)).uniform_(-1, 1)
        elif init_method == 'normal':
            init_a = torch.normal(1.0, 0.1, (n_params,))
            init_b = torch.normal(0.0, 0.1, (n_params,))
            
        self.rectify = rectify
        self.train_b = train_b
        self.dataset = dataset
        self.single_param = single_param
        self.positive_a = positive_a
        
        # Initialize learnable parameters
        if positive_a:
            # Use log for a to ensure it stays positive
            self.a_raw = nn.Parameter(torch.log(torch.clamp(init_a, min=1e-6)))
            self.a_raw.is_special_param = True
        else:
            self.a = nn.Parameter(init_a)
            self.a.is_special_param = True
        
        if self.train_b:
            self.b = nn.Parameter(init_b)
            self.b.is_special_param = True
        else:
            self.register_buffer('b', init_b)
                
        self.indim = indim

    def forward(self, x):
        # For colored datasets, expand parameters to match input dimensions
        if self.single_param:
            # Use the same parameters for all pixels
            if self.positive_a:
                a_expanded = self.a_raw.exp().expand(self.indim)
            else:
                a_expanded = self.a.expand(self.indim)
            if self.train_b:
                b_expanded = self.b.expand(self.indim)
            else:
                b_expanded = self.b.expand(self.indim)
        else:
            if self.positive_a:
                a_expanded = self.a_raw.exp()
            else:
                a_expanded = self.a
            b_expanded = self.b
            
        # Simple cubic transformation: ax^3 + b
        y = a_expanded * x**3 + b_expanded
        return torch.clamp(y, min=0) if self.rectify else y

class AdaptiveClassic_tuner_valuewise(nn.Module):
    '''
    Adaptive classical activation functions with 2 learnable parameters
    Based on AdaptiveClassic from neurons.py
    1. Adaptive ReLU: f(x)=max(0,ax+b)
    2. Adaptive LeakyReLU: f(x)=max(0,ax+b)+0.01*min(0,ax+b)
    3. Adaptive Sigmoid: f(x)=1/(1+exp(-ax-b))
    4. Adaptive tanh: f(x)=tanh(ax+b)
    a should be positive number, and no limit on b
    from <Adaptive Activation Functions for Deep Learning-based Power Flow Analysis>
    
    '''
    def __init__(self, indim, train_b, init_method, dataset, mode='ReLU', rectify=False, single_param=False, adaptive_minmax='none', positive_a=True):
        super().__init__()
        assert mode in ['ReLU','LeakyReLU','Sigmoid','tanh'], "mode should be in ['ReLU','LeakyReLU','Sigmoid','tanh']"
        assert init_method in ['define','uniform','normal']
        assert adaptive_minmax in ['none', 'symmetric', 'asymmetric'], "adaptive_minmax should be in ['none', 'symmetric', 'asymmetric']"
        
        self.mode = mode
        self.rectify = rectify
        self.train_b = train_b
        self.dataset = dataset
        self.single_param = single_param
        self.adaptive_minmax = adaptive_minmax
        self.positive_a = positive_a
        
        # For colored images, we want one function per value
        if single_param:
            n_params = 1  # Single parameter for all pixels
        else:
            n_params = indim
            
        # Initialize parameters based on init_method
        if init_method == 'define':
            init_a = torch.ones((n_params,))
            init_b = torch.zeros((n_params,))
        elif init_method == 'uniform':
            init_a = torch.empty((n_params,)).uniform_(0.0, 1.0)  # Positive values around 1
            init_b = torch.empty((n_params,)).uniform_(-1, 1)
        elif init_method == 'normal':
            init_a = torch.normal(1.0, 0.2, (n_params,))
            init_a = torch.clamp(init_a, min=0.1)  # Ensure positive
            init_b = torch.normal(0.0, 0.1, (n_params,))
            
        # Initialize learnable parameters
        if positive_a:
            # Use log for a to ensure it stays positive
            self.a_raw = nn.Parameter(torch.log(init_a))
        else:
            # Use a directly without transformation
            self.a_raw = nn.Parameter(init_a)
        self.a_raw.is_special_param = True
        
        if self.train_b:
            self.b = nn.Parameter(init_b)
            self.b.is_special_param = True
        else:
            self.register_buffer('b', init_b)
                
        self.indim = indim

    def forward(self, x):
        # For colored datasets, expand parameters to match input dimensions
        if self.single_param:
            # Use the same parameters for all pixels
            a_raw_expanded = self.a_raw.expand(self.indim)
            if self.train_b:
                b_expanded = self.b.expand(self.indim)
            else:
                b_expanded = self.b.expand(self.indim)
        else:
            a_raw_expanded = self.a_raw
            b_expanded = self.b
            
        # Convert a based on positive_a setting
        if self.positive_a:
            a = a_raw_expanded.exp()
        else:
            a = a_raw_expanded
            
        # Get min and max values for adaptive minmax
        if self.adaptive_minmax == 'symmetric':
            # Use symmetric bounds based on absolute maximum
            xmax = x.detach().abs().max()
        elif self.adaptive_minmax == 'asymmetric':
            # Use actual min and max values
            xmax = x.detach().max()
            xmin = x.detach().min()
        # For 'none', we don't need xmin/xmax
        
        # Apply the selected activation function
        if self.mode == 'ReLU':
            output = torch.relu(a * x + b_expanded)
        elif self.mode == 'LeakyReLU':
            output = torch.nn.functional.leaky_relu(a * x + b_expanded, negative_slope=0.01)
        elif self.mode == 'Sigmoid':
            if self.adaptive_minmax == 'none':
                output = torch.sigmoid(a * x + b_expanded)
            elif self.adaptive_minmax == 'symmetric':
                # Map [0,1] to [-xmax, xmax]
                output = 2*xmax*(torch.sigmoid(a * x + b_expanded) - 0.5)
            elif self.adaptive_minmax == 'asymmetric':
                # Map [0,1] to [xmin, xmax]
                output = xmin + (xmax - xmin) * torch.sigmoid(a * x + b_expanded)
        elif self.mode == 'tanh':
            if self.adaptive_minmax == 'none':
                output = torch.tanh(a * x + b_expanded)
            elif self.adaptive_minmax == 'symmetric':
                # Map [-1,1] to [-xmax, xmax] 
                output = xmax * torch.tanh(a * x + b_expanded)
            elif self.adaptive_minmax == 'asymmetric':
                # Map [-1,1] to [xmin, xmax]
                output = xmin + (xmax - xmin) * (torch.tanh(a * x + b_expanded) + 1) / 2
        
        return torch.clamp(output, min=0) if self.rectify else output
    

class SReLU(nn.Module):
    """
    S-shaped Rectified Linear Unit (SReLU) activation function.
    
    SReLU is defined as:
    - f(x) = t_left + a_left * (x - t_left) for x <= t_left
    - f(x) = x for t_left < x < t_right  
    - f(x) = t_right + a_right * (x - t_right) for x >= t_right
    """

    def __init__(self, rectify=False, neuronwise=False, num_neurons=1, init_method='uniform', positive_slopes=False):
        super(SReLU, self).__init__()
        
        self.neuronwise = neuronwise
        self.rectify = rectify
        self.positive_slopes = positive_slopes
        self.num_neurons = num_neurons if neuronwise else 1
        
        # Initialize parameters based on init_method
        if init_method == 'define':
            t_left_init = 0.0
            a_left_init = 0.5  # midpoint of U(0,1)
            t_right_init = 2.5  # midpoint of U(0,5)
            a_right_init = 1.0
        elif init_method == 'uniform':
            if neuronwise:
                t_left_init = torch.zeros(self.num_neurons)
                a_left_init = torch.empty(self.num_neurons).uniform_(0, 1)
                t_right_init = torch.empty(self.num_neurons).uniform_(0, 5)
                a_right_init = torch.ones(self.num_neurons)
            else:
                t_left_init = 0.0
                a_left_init = torch.empty(1).uniform_(0, 1).item()
                t_right_init = torch.empty(1).uniform_(0, 5).item()
                a_right_init = 1.0
        else:
            raise ValueError(f"Invalid init_method: {init_method}. Must be 'define' or 'uniform'.")
        
        # Initialize parameters
        if neuronwise:
            if isinstance(t_left_init, torch.Tensor):
                self.t_left = nn.Parameter(t_left_init)
                if positive_slopes:
                    self.a_left_raw = nn.Parameter(torch.log(a_left_init))
                    self.a_right_raw = nn.Parameter(torch.log(a_right_init))
                else:
                    self.a_left = nn.Parameter(a_left_init)
                    self.a_right = nn.Parameter(a_right_init)
                self.t_right = nn.Parameter(t_right_init)
            else:
                self.t_left = nn.Parameter(torch.full((self.num_neurons,), t_left_init))
                if positive_slopes:
                    self.a_left_raw = nn.Parameter(torch.log(torch.full((self.num_neurons,), a_left_init)))
                    self.a_right_raw = nn.Parameter(torch.log(torch.full((self.num_neurons,), a_right_init)))
                else:
                    self.a_left = nn.Parameter(torch.full((self.num_neurons,), a_left_init))
                    self.a_right = nn.Parameter(torch.full((self.num_neurons,), a_right_init))
                self.t_right = nn.Parameter(torch.full((self.num_neurons,), t_right_init))
        else:
            self.t_left = nn.Parameter(torch.tensor(t_left_init))
            if positive_slopes:
                self.a_left_raw = nn.Parameter(torch.log(torch.tensor(a_left_init)))
                self.a_right_raw = nn.Parameter(torch.log(torch.tensor(a_right_init)))
            else:
                self.a_left = nn.Parameter(torch.tensor(a_left_init))
                self.a_right = nn.Parameter(torch.tensor(a_right_init))
            self.t_right = nn.Parameter(torch.tensor(t_right_init))
        
        self.t_left.is_special_param = True
        if positive_slopes:
            self.a_left_raw.is_special_param = True
            self.a_right_raw.is_special_param = True
        else:
            self.a_left.is_special_param = True
            self.a_right.is_special_param = True
        self.t_right.is_special_param = True
        
    def forward(self, x):
        # Ensure t_right > t_left
        t_right_actual = self.t_left + torch.abs(self.t_right)
        
        # Get actual slope values
        if self.positive_slopes:
            a_left_actual = self.a_left_raw.exp()
            a_right_actual = self.a_right_raw.exp()
        else:
            a_left_actual = self.a_left
            a_right_actual = self.a_right
        
        # Use nested torch.where for three-way conditional
        output = torch.where(x <= self.t_left, 
                        self.t_left + a_left_actual * (x - self.t_left),
                        torch.where(x >= t_right_actual,
                                    t_right_actual + a_right_actual * (x - t_right_actual),
                                    x))
        
        return torch.clamp(output, min=0) if self.rectify else output
#%% flexible CM-GLLF
# Code for universal CM-GLLF function
def compute_gllf_gradient(x,xmin,xmax,yL,yR,mu,I,mode,offset=1e-6):
    # assert mode in ['x','mu','I']
    if mode == 'x':
        grad_x = ((yL - yR)*((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(
            x - xmin)**2 - torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))/(x - xmin) + (
                torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(1/mu - 2)*(x - xmax))/(
                    (x - xmin)*(xmax - xmin))))/((torch.exp((1/mu - 2)*(I - (x - xmin)/(
                        xmax - xmin)))*(x - xmax))/(x - xmin) - 1)**2
        return grad_x
    elif mode == 'x_offset':
        grad_x = ((yL - yR)*((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(
            offset + x - xmin)**2 - torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))/(offset + x - xmin) + (
                torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(1/mu - 2)*(x - xmax))/(
                    (xmax - xmin)*(offset + x - xmin))))/((torch.exp((1/mu - 2)*(I - (x - xmin)/(
                        xmax - xmin)))*(x - xmax))/(offset + x - xmin) - 1)**2
        return grad_x
    elif mode=='mu':
        grad_mu = (torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax)*(
            yL - yR)*(I - (x - xmin)/(xmax - xmin)))/(mu**2*((torch.exp((1/mu - 2)*(
                I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(x - xmin) - 1)**2*(x - xmin))
        return grad_mu
    elif mode=='mu_offset':
        grad_mu = (torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax)*(
            yL - yR)*(I - (x - xmin)/(xmax - xmin)))/(mu**2*((torch.exp((1/mu - 2)*(
                I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(offset + x - xmin) - 1)**2*(offset + x - xmin))
        return grad_mu
    elif mode=='I':
        grad_I = -(torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(1/mu - 2)*(
            x - xmax)*(yL - yR))/(((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(
                x - xmax))/(x - xmin) - 1)**2*(x - xmin))    
        return grad_I    
    elif mode=='I_offset':
        grad_I = -(torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(1/mu - 2)*(
            x - xmax)*(yL - yR))/(((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(
                x - xmax))/(offset + x - xmin) - 1)**2*(offset + x - xmin))
        return grad_I
    elif mode=='xmin_offset':
        grad_xmin = -(((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(offset + x - xmin)**2 + (torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(1/mu - 2)*(1/(xmax - xmin) - (x - xmin)/(xmax - xmin)**2)*(x - xmax))/(offset + x - xmin))*(yL - yR))/((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(offset + x - xmin) - 1)**2
        return grad_xmin
    elif mode=='xmax_offset':
        grad_xmax = ((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))/(offset + x - xmin) - (torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(1/mu - 2)*(x - xmax)*(x - xmin))/((xmax - xmin)**2*(offset + x - xmin)))*(yL - yR))/((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(offset + x - xmin) - 1)**2
        return grad_xmax
    elif mode=='yL_offset':
        grad_yL = 1/((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(offset + x - xmin) - 1) + 1
        return grad_yL
    elif mode=='yR_offset':
        grad_yR = -1/((torch.exp((1/mu - 2)*(I - (x - xmin)/(xmax - xmin)))*(x - xmax))/(offset + x - xmin) - 1)
        return grad_yR    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'x', 'mu', or 'I'.")


def interp1_nearest(xs_sorted, ys_sorted, x):
    """
    PyTorch equivalent of MATLAB's interp1 with 'nearest' and 'extrap'
    
    Args:
        xs_sorted: sorted x coordinates (1D tensor)
        ys_sorted: corresponding y values (1D tensor) 
        x: query points (tensor of any shape)
    
    Returns:
        interpolated values with same shape as x
    """

    # Ensure tensors are contiguous to avoid the warning
    xs_sorted = xs_sorted.contiguous()
    x = x.contiguous()

    # Find insertion points
    idx = torch.searchsorted(xs_sorted, x, right=True)
    
    # Handle boundary cases for extrapolation
    idx = torch.clamp(idx, 0, len(xs_sorted) - 1)
    
    # For points exactly on grid, searchsorted gives the next index
    # For nearest neighbor, we need to check which is closer
    mask = idx > 0
    left_dist = torch.where(mask, torch.abs(x - xs_sorted[idx - 1]), float('inf'))
    right_dist = torch.abs(x - xs_sorted[idx])
    
    # Use left point if it's closer
    use_left = (left_dist < right_dist) & mask
    idx = torch.where(use_left, idx - 1, idx)
    
    return ys_sorted[idx]


# def interp1_nearest_vectorized(xs, ys, x):
#     """
#     Vectorized PyTorch equivalent of MATLAB's interp1 with 'nearest' and 'extrap'
    
#     Args:
#         xs: sorted x coordinates (2D tensor: [n_points, n_neurons])
#         ys: corresponding y values (1D tensor: [n_points])  
#         x: query points (2D tensor: [batch_size, n_neurons])
    
#     Returns:
#         interpolated values with same shape as x
#     """
#     # xs: [n_points, n_neurons], x: [batch_size, n_neurons]
#     # We need to find nearest neighbors for each neuron separately but vectorized
    
#     # batch_size, n_neurons = x.shape
#     # n_points = xs.shape[0]
    
#     # Expand dimensions for broadcasting
#     # xs: [n_points, 1, n_neurons], x: [1, batch_size, n_neurons]
#     xs_expanded = xs.unsqueeze(1)  # [n_points, 1, n_neurons]
#     x_expanded = x.unsqueeze(0)    # [1, batch_size, n_neurons]
    
#     # Compute distances: [n_points, batch_size, n_neurons]
#     distances = torch.abs(xs_expanded - x_expanded)
    
#     # Find indices of minimum distances along the first dimension
#     # min_indices: [batch_size, n_neurons]
#     min_indices = torch.argmin(distances, dim=0)
    
#     # Use advanced indexing to get the corresponding y values
#     # ys is [n_points], min_indices is [batch_size, n_neurons]
#     result = ys[min_indices]
    
#     return result


def assert_sorted(tensor):
    """Assert that tensor is sorted in ascending order"""
    assert torch.all(tensor[1:] >= tensor[:-1]), "Tensor is not sorted in ascending order"


class CM_GLLF_tuner_flexible(nn.Module):
    '''
    CM-GLLF function using the flexible compute_gllf implementation
    Similar to CM_GLLF_tuner_valuewise_restrict but uses compute_gllf for neuron-wise operations
    In logistic range, mu is in (0,0.5), I is in (0,1)
    '''
    def __init__(self, indim, train_I, init_method, dataset,
                  minmax_setting='old',mode='logistic', rectify=False, offset=1e-6):
        super().__init__()
        self.minmax_setting = minmax_setting
        self.mode = mode
        self.offset = offset
            
        n_params = indim
        
        if init_method == 'define':
            if mode == 'logistic':
                init_mu = torch.full((n_params,), 0.25)
            elif mode == 'logit':
                init_mu = torch.full((n_params,), 0.75)
            elif mode == 'all':
                init_mu = torch.full((n_params,), 0.5)
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'logistic', 'logit', or 'all'.")
            init_I = torch.full((n_params,), 0.5)
        elif init_method == 'uniform':
            if mode == 'logistic':
                uniform_L, uniform_R = 0.125, 0.375
            elif mode == 'logit':
                uniform_L, uniform_R = 0.625, 0.875
            elif mode == 'all':
                uniform_L, uniform_R = 0.25, 0.75
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'logistic', 'logit', or 'all'.")
            init_mu = torch.empty((n_params,)).uniform_(uniform_L, uniform_R)
            if train_I:
                # init_I = torch.empty((n_params,)).uniform_(0.25, 0.75)
                init_I = torch.empty((n_params,)).uniform_(0.0, 1.0)
            else:
                init_I = torch.full((n_params,), 0.5)   
        else:
            raise ValueError(f"Invalid init_method: {init_method}. Must be 'define', 'uniform', 'uniform_wide', or 'normal'.")
            
        self.rectify = rectify
        self.train_I = train_I
        self.dataset = dataset
        
        # Initialize learnable parameters
        if mode == 'logistic':
            # For logistic mode, constrain mu to (0, 0.5)
            self.mu_raw = nn.Parameter(torch.logit(2 * init_mu))
        elif mode == 'logit':
            # For logit mode, constrain mu to (0.5, 1)
            self.mu_raw = nn.Parameter(torch.logit(2 * (init_mu - 0.5)))
        elif mode == 'all':
            # For all mode, mu can be in (0, 1)
            self.mu_raw = nn.Parameter(torch.logit(init_mu))
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'logistic', 'logit', or 'all'.")
            
        self.mu_raw.is_special_param = True
        
        if self.train_I:
            self.I_raw = nn.Parameter(torch.logit(init_I))
            self.I_raw.is_special_param = True
        else:
            self.register_buffer('I_raw', torch.logit(init_I))
            
        self.indim = indim

    def forward(self, x):
        # Transform parameters based on mode
        if self.mode == 'logistic':
            mu = 0.5 * torch.sigmoid(self.mu_raw)
        elif self.mode == 'logit':
            mu = 0.5 + 0.5 * torch.sigmoid(self.mu_raw)
        elif self.mode == 'all':
            mu = torch.sigmoid(self.mu_raw)
            
        I = torch.sigmoid(self.I_raw)
        
        # Use the flexible compute_gllf function
        return compute_gllf.apply(x, mu, I, self.rectify, self.minmax_setting, self.mode, self.dataset, self.offset)


class compute_gllf(Function):
    '''
    Each neuron learns a unique CM-GLLF curve
    '''
    @staticmethod
    def forward(ctx, x, mu, I, rectify, minmax_setting, mode, dataset, offset=1e-6):
        '''
        x: for MLP (batch_size,num_neuron)
           for CNN (batch_size,num_channel,height,width)
        mu: 1D tensor of size (num_neuron,) or (1,num_channel,1,1)
        I: 1D tensor of size (num_neuron,) or (1,num_channel,1,1)
        rectify(bool): if y values less than 0 should be set to 0
        mode: 3 modes for CM-GLLF and 1 for ReLU
            logistic: search the range of 0<Mu<=0.5
            logit: search the range of 0.5=<Mu<1
            all: mu not limited
        '''
        if minmax_setting == 'old':
            xmax = x.abs().max()
            xmin = -xmax
            yL = xmin
            yR = xmax        
        elif minmax_setting == 'minmax_mapping':
            xmin = x.detach().min()
            xmax = x.detach().max()
            yL = xmin
            yR = xmax
        elif minmax_setting == 'minmax_pn1':
            xmin = x.detach().min()
            xmax = x.detach().max()
            yL = -1
            yR = 1    
        elif minmax_setting == 'minmax_pn2':
            xmin = x.detach().min()
            xmax = x.detach().max()
            yL = -2
            yR = 2               
        elif minmax_setting == 'minmax_dataset':
            if dataset == 'CIFAR10':
                xmin = min([-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616])
                xmax = max([(1-0.4914)/0.2470, (1-0.4822)/0.2435, (1-0.4465)/0.2616])
                yL = xmin
                yR = xmax
            elif dataset == 'CIFAR100':
                xmin = min([-0.5071/0.2675, -0.4867/0.2563, -0.4409/0.2761])
                xmax = max([(1-0.5071)/0.2675, (1-0.4867)/0.2563, (1-0.4409)/0.2761])
                yL = xmin
                yR = xmax   
            else:
                raise ValueError(f"Invalid dataset: {dataset}. Must be 'CIFAR10' or 'CIFAR100'.")               
        else:
            raise ValueError(f"Invalid minmax_setting: {minmax_setting}. Must be 'old' or 'minmax_mapping'.")
        # calculate the output value
        # For logistic phase (0<=mu<=1)
        if mode in ['logistic','all']:
            y_logistic = yL + (yR-yL) / (1 + (xmax-x)/(x-xmin+offset)*torch.exp((2-1/mu)*((x-xmin)/(xmax-xmin)-I)))
        if mode in ['logit','all']:
            # Process each neuron separately for proper interpolation
            if x.dim() == 2:  # MLP case: (batch_size, num_neurons)
                if False:
                    ys = torch.linspace(yL, yR, 5000, device=x.device)
                    mu_T = torch.squeeze(mu)
                    I_T = torch.squeeze(I)
                    
                    # Create corresponding x values for the y grid using inverse CM-GLLF formula
                    xs = xmin + (xmax-xmin) / (1 + (yR-ys.unsqueeze(-1))/(ys.unsqueeze(-1)-yL+offset)*torch.exp((2-1/(1-mu_T.unsqueeze(0)))*((ys.unsqueeze(-1)-yL)/(yR-yL)-I_T.unsqueeze(0))))
                    
                    y_logit = torch.zeros_like(x)                        
                    for neuron_idx in range(x.shape[1]):
                    
                        x_neuron = x[:, neuron_idx]
                        xs_neuron = xs[:, neuron_idx]
                        # Use interp1_nearest for efficient nearest neighbor interpolation
                        y_logit[:, neuron_idx] = interp1_nearest(xs_neuron, ys, x_neuron)
                else:
                    # y_logit = interp1_nearest_vectorized(xs, ys, x)
                    ys = torch.linspace(yL, yR, 1000, device=x.device).unsqueeze(-1)
                    mu_T = torch.squeeze(mu).unsqueeze(0)
                    I_T = torch.squeeze(I).unsqueeze(0)  
                    # create xs:(1000,N) for each x, 1000 candidate x value, choose nearest one
                    xs = xmin + (xmax-xmin) / (1 + (yR-ys)/(ys-yL+offset)*torch.exp((2-1/(1-mu_T))*((ys-yL)/(yR-yL)-I_T)))
                    y_logit = torch.zeros(x.shape, device=x.device)
                    xI = xmin + I * (xmax - xmin)
                    # mask_left = x < xI
                    if x.dim()==2:
                        idx_left = torch.argmin(torch.abs(x.unsqueeze(0) - xs.unsqueeze(1)), dim=0)
                    elif x.dim()==4:
                        idx_left = torch.argmin(torch.abs(x.unsqueeze(0) - xs.unsqueeze(-1).unsqueeze(-1).unsqueeze(1)), dim=0)
                    else:
                        raise ValueError('Shape of x is neither 2 nor 4!')
                    # y_logit[mask_left] = torch.squeeze(ys)[idx_left[mask_left]]
                    y_logit[x < xI] = torch.squeeze(ys)[idx_left[x < xI]]
                    # mask_right = x >= xI
                    if x.dim()==2:
                        idx_right = torch.argmin(torch.abs(x.unsqueeze(0) - torch.flip(xs, dims=[0]).unsqueeze(1)), dim=0)
                    elif x.dim()==4:
                        idx_right = torch.argmin(torch.abs(x.unsqueeze(0) - torch.flip(xs, dims=[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(1)), dim=0)
                    else:
                        raise ValueError('Shape of x is neither 2 nor 4!')            
                    # ys_flipped = torch.flip(ys, dims=[0])
                    # y_logit[mask_right] = torch.squeeze(ys_flipped)[idx_right[mask_right]]
                    y_logit[x >= xI] = torch.squeeze(torch.flip(ys, dims=[0]))[idx_right[x >= xI]]                        
                    
            # elif x.dim() == 4:  # CNN case: (batch_size, channels, height, width)
            #     for channel_idx in range(x.shape[1]):
            #         x_channel = x[:, channel_idx, :, :]
            #         xs_channel = xs[:, channel_idx]
                    
            #         # Sort xs and corresponding ys for interpolation
            #         sorted_indices = torch.argsort(xs_channel)
            #         xs_sorted = xs_channel[sorted_indices]
            #         ys_sorted = ys[sorted_indices]
                    
            #         # Use interp1_nearest for efficient nearest neighbor interpolation
            #         y_logit[:, channel_idx, :, :] = interp1_nearest(xs_sorted, ys_sorted, x_channel)
            else:
                raise ValueError('Shape of x is neither 2 nor 4!')
        if mode == 'all':
            # assign CM-GLLF values based on the cases
            y = torch.where((mu >= 0) & (mu <= 0.5),y_logistic,y_logit)   
        elif mode == 'logistic':
            y = y_logistic
        elif mode == 'logit':
            # this makes logit phase include ReLU
            y = y_logit
        else:
            raise ValueError("Mode should be in [all, logistic, logit]")
        # save the data for backward pass
        ctx.save_for_backward(x, y, mu, I,)
        ctx.scalars = (xmin, xmax, yL, yR, rectify, mode, offset)
        return torch.clamp(y, min=0) if rectify else y  
    @staticmethod
    def backward(ctx, grad_output):
        x, y, mu, I = ctx.saved_tensors
        xmin, xmax, yL, yR, rectify, mode, offset = ctx.scalars
        # check if mu and I needs gradient
        mu_need_grad = ctx.needs_input_grad[1]
        I_need_grad = ctx.needs_input_grad[2]
        if mode in ['logistic','all']:
            # for gradients of logistic phase (see ./matlab_utils/get_gllf_drv)
            grad_x_logistic = compute_gllf_gradient(x,xmin,xmax,yL,yR,mu,I,mode='x_offset',offset=offset)
            if mu_need_grad:
                grad_mu_logistic = compute_gllf_gradient(x,xmin,xmax,yL,yR,mu,I,mode='mu_offset',offset=offset)
            if I_need_grad:
                grad_I_logistic = compute_gllf_gradient(x,xmin,xmax,yL,yR,mu,I,mode='I_offset',offset=offset)
        if mode in ['logit','all']:
            # for gradients of logit phase
            # derivative of inverse function is 1/f'(f^(-1)(x))
            inv_grad_x_logit = compute_gllf_gradient(y,xmin,xmax,yL,yR,1-mu,I,mode='x_offset',offset=offset)
            grad_x_logit = 1.0 / inv_grad_x_logit
            # derivative of logit function respect to mu and I
            if mu_need_grad:
                grad_mu_logit = (-1) * compute_gllf_gradient(y,xmin,xmax,yL,yR,1-mu,I,mode='mu_offset',offset=offset) / inv_grad_x_logit
            if I_need_grad:
                grad_I_logit = (-1) * compute_gllf_gradient(y,xmin,xmax,yL,yR,1-mu,I,mode='I_offset',offset=offset) / inv_grad_x_logit
        # merge the gradients, assign based on logistic and logit
        if mode == 'all':
            grad_x = torch.where((mu >= 0) & (mu <= 0.5), grad_x_logistic, grad_x_logit)
            if ctx.needs_input_grad[1]:  
                grad_mu = torch.where((mu >= 0) & (mu <= 0.5), grad_mu_logistic, grad_mu_logit)
            else:
                grad_mu = None
            if ctx.needs_input_grad[2]:  
                grad_I = torch.where((mu >= 0) & (mu <= 0.5), grad_I_logistic, grad_I_logit)
            else:
                grad_I = None         
        elif mode == 'logistic':
            grad_x = grad_x_logistic
            if ctx.needs_input_grad[1]:  
                grad_mu = grad_mu_logistic
            else:
                grad_mu = None
            if ctx.needs_input_grad[2]:  
                grad_I = grad_I_logistic
            else:
                grad_I = None                    
        elif mode == 'logit':
            grad_x = grad_x_logit
            if ctx.needs_input_grad[1]:  
                grad_mu = grad_mu_logit
            else:
                grad_mu = None
            if ctx.needs_input_grad[2]:  
                grad_I = grad_I_logit
            else:
                grad_I = None                     
        # if rectify, do:
        if rectify:
            grad_x[y<=0] = 0
            if ctx.needs_input_grad[1]:
                grad_mu[y<=0] = 0
            if ctx.needs_input_grad[2]:
                grad_I[y<=0] = 0            
        drv_x = drv_mu = drv_I = None
        if ctx.needs_input_grad[0]:
            drv_x = grad_output * grad_x
        if ctx.needs_input_grad[1]:
            if x.dim()==2:
                drv_mu = (grad_output * grad_mu).sum(dim=0)
            elif x.dim()==4:
                drv_mu = torch.sum(grad_output * grad_mu, dim=(0, 2, 3), keepdim=True)
        if ctx.needs_input_grad[2]:
            if x.dim()==2:
                drv_I = (grad_output * grad_I).sum(dim=0)
            elif x.dim()==4:
                drv_I = torch.sum(grad_output * grad_I, dim=(0, 2, 3), keepdim=True)     
        return drv_x, drv_mu, drv_I, None, None, None, None, None


class compute_gllf_trainable_minmax(Function):
    '''
    CM-GLLF function with trainable min/max parameters
    Handles out-of-range inputs by setting them and their gradients to 0
    '''
    @staticmethod
    def forward(ctx, x, mu, I, xmin, xmax, yL, yR, rectify, mode, offset=1e-6):
        '''
        x: input tensor
        mu, I, xmin, xmax, yL, yR: trainable parameters
        rectify: whether to apply ReLU
        mode: 'logistic', 'logit', or 'all'
        offset: numerical stability offset
        '''
        # Create mask for inputs within valid range [xmin, xmax]
        in_range_mask = (x >= xmin) & (x <= xmax)
        
        # Calculate CM-GLLF output
        if mode in ['logistic','all']:
            y_logistic = yL + (yR-yL) / (1 + (xmax-x)/(x-xmin+offset)*torch.exp((2-1/mu)*((x-xmin)/(xmax-xmin)-I)))
        
        if mode in ['logit','all']:
            # For logit phase, use similar approach as in original compute_gllf
            ys = torch.linspace(yL.min().item(), yR.max().item(), 1000, device=x.device).unsqueeze(-1)
            mu_T = torch.squeeze(mu).unsqueeze(0)
            I_T = torch.squeeze(I).unsqueeze(0)
            xmin_T = torch.squeeze(xmin).unsqueeze(0)
            xmax_T = torch.squeeze(xmax).unsqueeze(0)
            yL_T = torch.squeeze(yL).unsqueeze(0)
            yR_T = torch.squeeze(yR).unsqueeze(0)
            
            # Create xs for interpolation
            xs = xmin_T + (xmax_T-xmin_T) / (1 + (yR_T-ys)/(ys-yL_T+offset)*torch.exp((2-1/(1-mu_T))*((ys-yL_T)/(yR_T-yL_T)-I_T)))
            y_logit = torch.zeros(x.shape, device=x.device)
            xI = xmin + I * (xmax - xmin)
            
            if x.dim()==2:
                idx_left = torch.argmin(torch.abs(x.unsqueeze(0) - xs.unsqueeze(1)), dim=0)
                idx_right = torch.argmin(torch.abs(x.unsqueeze(0) - torch.flip(xs, dims=[0]).unsqueeze(1)), dim=0)
            elif x.dim()==4:
                idx_left = torch.argmin(torch.abs(x.unsqueeze(0) - xs.unsqueeze(-1).unsqueeze(-1).unsqueeze(1)), dim=0)
                idx_right = torch.argmin(torch.abs(x.unsqueeze(0) - torch.flip(xs, dims=[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(1)), dim=0)
            else:
                raise ValueError('Shape of x is neither 2 nor 4!')
                
            y_logit[x < xI] = torch.squeeze(ys)[idx_left[x < xI]]
            y_logit[x >= xI] = torch.squeeze(torch.flip(ys, dims=[0]))[idx_right[x >= xI]]
        
        # Select output based on mode
        if mode == 'all':
            y = torch.where((mu >= 0) & (mu <= 0.5), y_logistic, y_logit)   
        elif mode == 'logistic':
            y = y_logistic
        elif mode == 'logit':
            y = y_logit
        else:
            raise ValueError("Mode should be in [all, logistic, logit]")
        
        # Apply out-of-range masking to output
        y = torch.where(in_range_mask, y, torch.zeros_like(y))
        
        # Save for backward pass
        ctx.save_for_backward(x, y, mu, I, xmin, xmax, yL, yR, in_range_mask)
        ctx.scalars = (rectify, mode, offset)
        
        return torch.clamp(y, min=0) if rectify else y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, mu, I, xmin, xmax, yL, yR, in_range_mask = ctx.saved_tensors
        rectify, mode, offset = ctx.scalars
        
        # Initialize gradients
        grad_x = grad_mu = grad_I = grad_xmin = grad_xmax = grad_yL = grad_yR = None
        
        # Only compute gradients for parameters that need them
        x_need_grad = ctx.needs_input_grad[0]
        mu_need_grad = ctx.needs_input_grad[1]
        I_need_grad = ctx.needs_input_grad[2]
        xmin_need_grad = ctx.needs_input_grad[3]
        xmax_need_grad = ctx.needs_input_grad[4]
        yL_need_grad = ctx.needs_input_grad[5]
        yR_need_grad = ctx.needs_input_grad[6]
        
        if mode in ['logistic','all']:
            # Gradients for logistic phase
            if x_need_grad:
                grad_x_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='x_offset', offset=offset)
            if mu_need_grad:
                grad_mu_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='mu_offset', offset=offset)
            if I_need_grad:
                grad_I_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='I_offset', offset=offset)
            if xmin_need_grad:
                grad_xmin_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='xmin_offset', offset=offset)
            if xmax_need_grad:
                grad_xmax_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='xmax_offset', offset=offset)
            if yL_need_grad:
                grad_yL_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='yL_offset', offset=offset)
            if yR_need_grad:
                grad_yR_logistic = compute_gllf_gradient(x, xmin, xmax, yL, yR, mu, I, mode='yR_offset', offset=offset)

        if mode in ['logit','all']:
            # for gradients of logit phase
            # derivative of inverse function is 1/f'(f^(-1)(x))
            inv_grad_x_logit = compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='x_offset', offset=offset)
            if x_need_grad:
                grad_x_logit = 1.0 / inv_grad_x_logit
            if mu_need_grad:
                grad_mu_logit = (-1) * compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='mu_offset', offset=offset) / inv_grad_x_logit
            if I_need_grad:
                grad_I_logit = (-1) * compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='I_offset', offset=offset) / inv_grad_x_logit
            if xmin_need_grad:
                grad_xmin_logit = (-1) * compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='xmin_offset', offset=offset) / inv_grad_x_logit
            if xmax_need_grad:
                grad_xmax_logit = (-1) * compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='xmax_offset', offset=offset) / inv_grad_x_logit
            if yL_need_grad:
                grad_yL_logit = (-1) * compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='yL_offset', offset=offset) / inv_grad_x_logit
            if yR_need_grad:
                grad_yR_logit = (-1) * compute_gllf_gradient(y, xmin, xmax, yL, yR, 1-mu, I, mode='yR_offset', offset=offset) / inv_grad_x_logit

        # Merge gradients based on mode
        if mode == 'all':
            if x_need_grad:
                grad_x = torch.where((mu >= 0) & (mu <= 0.5), grad_x_logistic, grad_x_logit)
            if mu_need_grad:
                grad_mu = torch.where((mu >= 0) & (mu <= 0.5), grad_mu_logistic, grad_mu_logit)
            if I_need_grad:
                grad_I = torch.where((mu >= 0) & (mu <= 0.5), grad_I_logistic, grad_I_logit)
            if xmin_need_grad:
                grad_xmin = torch.where((mu >= 0) & (mu <= 0.5), grad_xmin_logistic, grad_xmin_logit)
            if xmax_need_grad:
                grad_xmax = torch.where((mu >= 0) & (mu <= 0.5), grad_xmax_logistic, grad_xmax_logit)
            if yL_need_grad:
                grad_yL = torch.where((mu >= 0) & (mu <= 0.5), grad_yL_logistic, grad_yL_logit)
            if yR_need_grad:
                grad_yR = torch.where((mu >= 0) & (mu <= 0.5), grad_yR_logistic, grad_yR_logit)
        elif mode == 'logistic':
            if x_need_grad:
                grad_x = grad_x_logistic
            if mu_need_grad:
                grad_mu = grad_mu_logistic
            if I_need_grad:
                grad_I = grad_I_logistic
            if xmin_need_grad:
                grad_xmin = grad_xmin_logistic
            if xmax_need_grad:
                grad_xmax = grad_xmax_logistic
            if yL_need_grad:
                grad_yL = grad_yL_logistic
            if yR_need_grad:
                grad_yR = grad_yR_logistic
        elif mode == 'logit':
            if x_need_grad:
                grad_x = grad_x_logit
            if mu_need_grad:
                grad_mu = grad_mu_logit
            if I_need_grad:
                grad_I = grad_I_logit
            if xmin_need_grad:
                grad_xmin = grad_xmin_logit
            if xmax_need_grad:
                grad_xmax = grad_xmax_logit
            if yL_need_grad:
                grad_yL = grad_yL_logit
            if yR_need_grad:
                grad_yR = grad_yR_logit

        # Apply out-of-range masking to all gradients
        if x_need_grad:
            grad_x = torch.where(in_range_mask, grad_x, torch.zeros_like(grad_x))
        if mu_need_grad:
            grad_mu = torch.where(in_range_mask, grad_mu, torch.zeros_like(grad_mu))
        if I_need_grad:
            grad_I = torch.where(in_range_mask, grad_I, torch.zeros_like(grad_I))
        if xmin_need_grad:
            grad_xmin = torch.where(in_range_mask, grad_xmin, torch.zeros_like(grad_xmin))
        if xmax_need_grad:
            grad_xmax = torch.where(in_range_mask, grad_xmax, torch.zeros_like(grad_xmax))
        if yL_need_grad:
            grad_yL = torch.where(in_range_mask, grad_yL, torch.zeros_like(grad_yL))
        if yR_need_grad:
            grad_yR = torch.where(in_range_mask, grad_yR, torch.zeros_like(grad_yR))

        # Apply rectification masking
        if rectify:
            if x_need_grad:
                grad_x = torch.where(y > 0, grad_x, torch.zeros_like(grad_x))
            if mu_need_grad:
                grad_mu = torch.where(y > 0, grad_mu, torch.zeros_like(grad_mu))
            if I_need_grad:
                grad_I = torch.where(y > 0, grad_I, torch.zeros_like(grad_I))
            if xmin_need_grad:
                grad_xmin = torch.where(y > 0, grad_xmin, torch.zeros_like(grad_xmin))
            if xmax_need_grad:
                grad_xmax = torch.where(y > 0, grad_xmax, torch.zeros_like(grad_xmax))
            if yL_need_grad:
                grad_yL = torch.where(y > 0, grad_yL, torch.zeros_like(grad_yL))
            if yR_need_grad:
                grad_yR = torch.where(y > 0, grad_yR, torch.zeros_like(grad_yR))

        # Compute final gradients with chain rule
        drv_x = drv_mu = drv_I = drv_xmin = drv_xmax = drv_yL = drv_yR = None
        
        if x_need_grad:
            drv_x = grad_output * grad_x
        if mu_need_grad:
            if x.dim() == 2:
                drv_mu = (grad_output * grad_mu).sum(dim=0)
            elif x.dim() == 4:
                drv_mu = torch.sum(grad_output * grad_mu, dim=(0, 2, 3), keepdim=True)
        if I_need_grad:
            if x.dim() == 2:
                drv_I = (grad_output * grad_I).sum(dim=0)
            elif x.dim() == 4:
                drv_I = torch.sum(grad_output * grad_I, dim=(0, 2, 3), keepdim=True)
        if xmin_need_grad:
            if x.dim() == 2:
                drv_xmin = (grad_output * grad_xmin).sum(dim=0)
            elif x.dim() == 4:
                drv_xmin = torch.sum(grad_output * grad_xmin, dim=(0, 2, 3), keepdim=True)
        if xmax_need_grad:
            if x.dim() == 2:
                drv_xmax = (grad_output * grad_xmax).sum(dim=0)
            elif x.dim() == 4:
                drv_xmax = torch.sum(grad_output * grad_xmax, dim=(0, 2, 3), keepdim=True)
        if yL_need_grad:
            if x.dim() == 2:
                drv_yL = (grad_output * grad_yL).sum(dim=0)
            elif x.dim() == 4:
                drv_yL = torch.sum(grad_output * grad_yL, dim=(0, 2, 3), keepdim=True)
        if yR_need_grad:
            if x.dim() == 2:
                drv_yR = (grad_output * grad_yR).sum(dim=0)
            elif x.dim() == 4:
                drv_yR = torch.sum(grad_output * grad_yR, dim=(0, 2, 3), keepdim=True)
        
        return drv_x, drv_mu, drv_I, drv_xmin, drv_xmax, drv_yL, drv_yR, None, None, None

