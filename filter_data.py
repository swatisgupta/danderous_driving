import plotly.plotly as py1
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy import signal

import numpy as np
import pandas as pd
import scipy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from statistics import median
from statistics import mean
from statistics import variance
from statistics import stdev
from sklearn.externals.six import StringIO  
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

from scipy import signal

from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

functions = {'Max': max, 'Avg': mean, 'Std': stdev, 'Mid': median }
plot=0

def get_kalman():
    ls_filter = KalmanFilter(dim_x=2, dim_z=1)
    ls_filter.x = np.array([0., 0.]) # intial states: memory pressure, memory pressure gradient
    ls_filter.F = np.array([[1., 1.], [0.,1.]]) # state transition matrix
    ls_filter.H = np.array([[1., 0.]]) # measurement function
    ls_filter.P *= 0. # initial uncertainty of states
    ls_filter.R = 1 # measurement noise (larger, smoothier)
    # refelcts uncertainly from higher-order unknow input (model noise)
    ls_filter.Q = Q_discrete_white_noise(dim=2, dt=0.5, var=0.2)
    return ls_filter

def apply_kalman(dataset, val_idx):
    kal_filter = get_kalman() 
    vals = dataset[val_idx].values.tolist()
    kal_out = []
    for x in vals:
        kal_filter.predict()
        kal_filter.update(x)
        fval = kal_filter.x[0]
        kal_out.append(fval)    
    n_lbl = val_idx + "_kalman"
    dataset[n_lbl] = kal_out
    return dataset

def compute_fn_sliding_window(dataset, val_idx, window):
    print("Computing Avg, Median, Std for " + val_idx)
    vals = dataset[val_idx].values.tolist()
    #labl = dataset[label_idx].values.tolist()
    for fn in functions.keys():
        lst = []
        k = 0
        flag = 0
        for i in range(len(vals)):
            if i != 0:
                flag = 0
                k = 0
            if flag == 0 and k < window - 1:    
                lst.append(vals[i])
                k = k + 1
            else:
                flag = 1
                cval = functions[fn](vals[i-k:i])
                lst.extend([float(cval)])
        n_lbl = val_idx + "_" + fn + "_" + str(window)
        dataset[n_lbl] = lst
    return dataset

def apply_lowess(data, feature):
    observations = list(data[feature])
    input_range = list(range(len(observations)))
    print(len(observations), len(input_range)) 
    filtered = lowess(observations, input_range, frac=0.05, return_sorted=False)
    lbl=feature + "_lowess"
    data[lbl] = filtered.tolist()    
    if plot == 1:
        nm = 'Lowess Filter' + "_" + feature
        fname='lowess-filter' + "_" + feature    
        plot_feature(new_signal, plot_name, fname)
    return data

def kalman_filter(data, feature):
    observations = data[feature]
    observations.head()
    n_samples = len(observations)
    initial_value_guess = [0, 1]
    delta_t = np.pi * 2 / n_samples
    stddev=0.1
    transition_matrix = [[1, delta_t], [-delta_t, 1]]
    transition_covariance = np.diag([0.2, 0.2]) ** 2
    observation_covariance = np.diag([stddev, stddev]) ** 2
    kf = KalmanFilter(
    initial_state_mean=initial_value_guess,
    initial_state_covariance=observation_covariance,
    observation_covariance=observation_covariance
    )
    pred_state, state_cov = kf.smooth(observations)
    lbl=feature + "_kalman"
    data[lbl] = pred_state    
    if plot == 1:
        nm = 'Kalman Filter' + "_" + feature
        fname='kalman-filter' + "_" + feature    
    return data



def plot_feature(data, plot_name, fname):

    trace1 = go.Scatter(
       x=list(range(len(list(data)))),
       y=data,
       mode='lines',
       name=plot_name
    )

    layout = go.Layout(
        title=plot_name
        #showlegend=True
    )

    trace_data = [trace1]
    fig = go.Figure(data=trace_data, layout=layout)
    py1.iplot(fig, filename=fname)


def butter_low_pass(data, feature, fc=0.2):
    observation = list(data[feature])
    b, a = signal.butter(3, fc, btype='lowpass', analog=False)
    new_signal = signal.filtfilt(b, a, observation)
  
    lbl=feature + "_blp"
    data[lbl] = new_signal    
    if plot == 1:
        nm = 'Butter Low-Pass Filter' + "_" + feature
        fname='butter-low-pass-filter' + "_" + feature    
        plot_feature(new_signal, plot_name, fname)
    return data

def low_pass(data, feature, fc=0.2, b=0.08):
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)
 
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    s = list(data[feature])
    new_signal = np.convolve(s, sinc_func, mode="same")
    lbl=feature + "_lp"
    data[lbl] = new_signal    
    if plot == 1:
        nm = 'Low-Pass Filter' + "_" + feature
        fname='fft-low-pass-filter' + "_" + feature    
        plot_feature(new_signal, plot_name, fname)
    return data

def high_pass(data, feature, fc=0.1, b=0.08):
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)
 
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = np.blackman(N)
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    # reverse function
    sinc_func = -sinc_func
    sinc_func[int((N - 1) / 2)] += 1

    s = list(data[feature])
    new_signal = np.convolve(s, sinc_func, mode = "same")
    lbl=feature + "_hp"
    data[lbl] = new_signal    
    if plot == 1:
        nm = 'High-Pass Filter' + "_" + feature
        fname='fft-high-pass-filter' + "_" + feature
        plot_feature(new_signal, plot_name, fname)
    return data

def band_pass(data, feature, fL=0.1, fH=0.3, b=0.08):
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
 
    # low-pass filter
    hlpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)
 
    # high-pass filter 
    hhpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((N - 1) / 2)] += 1
 
    h = np.convolve(hlpf, hhpf)
    s = list(data[feature])
    new_signal = np.convolve(s, h, mode="same")
    print(len(new_signal), len(s))
    lbl=feature + "_bp"
    data[lbl] = new_signal    
    if plot == 1:
        nm = 'Band-Pass Filter' + "_" + feature
        fname='fft-band-pass-filter' + "_" + feature
        plot_feature(new_signal, nm, fname)
    return data

