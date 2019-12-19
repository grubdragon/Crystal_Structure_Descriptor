import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def clamp_outliers(x, method='auto', auto_pct=0.05, auto_kind='both', clamp_above=None, clamp_below=None):
    if method=='auto': 
        if auto_kind == 'both':
            clamp_below = np.percentile(x, auto_pct/2)
            clamp_above = np.percentile(x, 1-auto_pct/2)
        if auto_kind == 'below':
            clamp_below = np.percentile(x, auto_pct)
        if auto_kind == 'above':
            clamp_above = np.percentile(x, 1-auto_pct)
        return np.clip(x, clamp_below, clamp_above), clamp_above, clamp_below
    elif method=='values':
        return np.clip(x, clamp_below, clamp_above)
    else:
        raise NoSuchMethodException
        
def sma(X, window=100):
    avg_kernel = np.ones((1,window))/window
    res = signal.convolve2d(X, avg_kernel, mode='valid')
    return res

def gma(X, window=100):
    res = gaussian_filter1d(X, window/5, axis=1, mode='mirror')
    return res

def window_mask(axis, mask_lower, mask_upper):
    mask = (axis>=mask_lower)*(axis<=mask_upper)
    return mask

def window_sum(X, window_mask):
    return np.sum(X*window_mask, axis=1)

def shuffle_data(X, y, seed=None):
    ind = np.arange(len(y))
    
    if seed:
        np.random.seed(seed)
        
    np.random.shuffle(ind)

    return X[ind], y[ind], ind

def split(X, y, train_pct=0.8):
    X_train = X[:int(len(y)*train_pct)]
    y_train = y[:int(len(y)*train_pct)]
    X_test = X[int(len(y)*train_pct):]
    y_test = y[int(len(y)*train_pct):]
    return X_train, y_train, X_test, y_test