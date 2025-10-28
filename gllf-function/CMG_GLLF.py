import numpy as np
def CMG_GLLF(x, mu, xmin, xmax, yL, yR, I=0.5, rectify=0):
    '''
    Cannistraci-Muscoloni-Gu model for the generalized logistic-logit function

    INPUT:
    ------
    x : array_like
        Numerical vector of the values at which the function is evaluated, xmin<=x<=xmax.
    
    mu : float
        Type-growth-rate parameter, 0<=mu<=1.
        mu=0:      step function 
        0<mu<0.5:  generalized logistic function
        mu=0.5:    linear function
        0.5<mu<1:  generalized logit function
        mu=1:      constant function 
    
    xmin : float
        Minimum x value allowed.
    
    xmax : float
        Maximum x value allowed.
    
    yL : float
        Value of the function at xmin.
    
    yR : float
        Value of the function at xmax.
    
    I : float, optional (default=0.5)
        Inflection parameter, 0<=I<=1.
        It represents the proportion of the x-axis range [xmin,xmax] at which the inflection occurs.
        In particular, the inflection point of the logistic-logit function is at coordinates:
        xI = xmin+I*(xmax-xmin)
        yI = yL+I*(yR-yL)
    
    rectify : int, optional (default=0)
        1 or 0 to indicate if the function should be rectified or not.

    OUTPUT:
    -------
    y : ndarray
        Numerical vector of the values of the function evaluated at x.
    '''
    
    # Convert x to numpy array if it isn't already
    x = np.asarray(x)
    
    # Input validation
    if not (0 <= mu <= 1):
        raise ValueError('mu must be between 0 and 1.')
    if xmax <= xmin:
        raise ValueError('xmax must be greater than xmin.')
    if np.any(x < xmin) or np.any(x > xmax):
        raise ValueError('x values must be between xmin and xmax.')
    if not (0 <= I <= 1):
        raise ValueError('I must be between 0 and 1.')
    if rectify not in [0, 1]:
        raise ValueError('rectify must be 0 or 1.')
    
    # Compute the function
    if mu == 1:
        # constant function
        y = np.full_like(x, yL + I * (yR - yL), dtype=float)
    elif mu == 0:
        # step function
        y = np.zeros_like(x, dtype=float)
        xI = xmin + I * (xmax - xmin)
        y[x < xI] = yL
        y[x > xI] = yR
        y[x == xI] = max(yL, yR)
    elif mu <= 0.5:
        # generalized logistic function (linear function for mu=0.5)
        y = yL + (yR - yL) / (1 + (xmax - x) / (x - xmin) * np.exp((2 - 1 / mu) * ((x - xmin) / (xmax - xmin) - I)))
    else:
        # generalized logit function (computational approximation)
        if yL < yR:
            ys = np.linspace(yL, yR, 1000)
            xs = xmin + (xmax - xmin) / (1 + (yR - ys) / (ys - yL) * np.exp((2 - 1 / (1 - mu)) * ((ys - yL) / (yR - yL) - I)))
        else:
            ys = np.linspace(yR, yL, 1000)
            xs = xmin + (xmax - xmin) / (1 + (yL - ys) / (ys - yR) * np.exp((2 - 1 / (1 - mu)) * ((ys - yR) / (yL - yR) - I)))
            ys = yR + yL - ys
        
        # for mu -> 1, there are cases when multiple y values collapse on the same x value
        # if (yL<yR & x<xI) | (yL>yR & x>=xI), take the lowest of these y values
        # if (yL<yR & x>=xI) | (yL>yR & x<xI), take the highest of these y values
        # this choice ensures to reach the extreme points [xmin,yL] and [xmax,yR]
        y = np.full_like(x, np.nan, dtype=float)
        xI = xmin + I * (xmax - xmin)
        mask = x < xI
        if np.any(mask):
            x_masked = x[mask].flatten()
            idx = np.argmin(np.abs(x_masked[:, np.newaxis] - xs[np.newaxis, :]), axis=1)
            y[mask] = ys[idx]
        
        mask = x >= xI
        if np.any(mask):
            x_masked = x[mask].flatten()
            xs_reversed = xs[::-1]
            idx = np.argmin(np.abs(x_masked[:, np.newaxis] - xs_reversed[np.newaxis, :]), axis=1)
            ys_reversed = ys[::-1]
            y[mask] = ys_reversed[idx]
    
    if rectify:
        y = np.maximum(y, 0)
    
    return y
