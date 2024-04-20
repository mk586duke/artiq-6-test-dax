import numpy as np
from scipy.optimize import curve_fit
from  oitg.fitting.FitBase import FitBase, FitParameters, FitError
import oitg.fitting as fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import h5py as h5py
from collections import defaultdict
from scipy.signal import correlate, filtfilt, butter
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema
from scipy.signal import welch
from scipy.fftpack import fft, fftfreq
from scipy.signal import blackman,boxcar,hanning
from scipy.optimize import differential_evolution

from scipy.signal import find_peaks
from scipy.signal import blackman,boxcar,hanning

def period_determination_FFT_global_min(x,y):

    #define amplitude
    a = np.sqrt( np.mean((y-np.mean(y))**2) ) * np.sqrt(2)

    #define offset
    y0 = np.mean(y)

    # Normalize the data
    y_normalized = y - np.mean(y)

    # Windowing
    window = boxcar(len(y_normalized))
    y_normalized *= window

    # Zero padding (if needed, probably ok as it is with a window)
    y_padded = np.pad(y_normalized, (0, 0))


    # Perform Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(y_padded)
    freqs = np.fft.fftfreq(len(x), x[1] - x[0])  # Frequency values corresponding to FFT result

    # Find the dominant frequency (corresponding to the period) using the maximum amplitude
    dominant_freq_index = np.argmax(np.abs(fft_result))
    dominant_period = 1 / np.abs(freqs[dominant_freq_index])

    # Define the sine function
    def sine_function(x, period, phase):
        return np.sin(2*np.pi/period * x + phase)*a

    # Use differential evolution optimization, for global minimum
    bounds_fit = [ (0.5*dominant_period, 2*dominant_period), (-np.pi, np.pi)]
    result = differential_evolution(lambda params: np.sum((y_normalized - sine_function(x, *params))**2), bounds_fit, maxiter=100000, tol = 1e-1)

    # Extract the fitted parameters
    period_fit, phase_fit = result.x

    return(period_fit, phase_fit, a, y0)

def fitting_function(x, p):

    y = p['a']*np.sin(p['x0'] + 2*np.pi*(x)/p['period'])
    y += p['y0']

    return y

def parameter_initialiser_fft_global_min(x, y, p):
    p['period'], p['x0'], p['a'], p['y0'] = period_determination_FFT_global_min(x,y)

# Sine with initialiser which extracts the initial period with
# an fft, only works when the x-axis is regularly spaced
sin_ffti_global = FitBase(['x0', 'y0', 'a', 'period'], fitting_function,
                        parameter_initialiser=parameter_initialiser_fft_global_min)