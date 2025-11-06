# -*-coding:utf-8 -*-
"""
@File    :   dslib.py
@Time    :   2025/10/15 15:50
@Author  :   Noé Schreckenberg
@Version :   2.1
@Contact :   schreckenberg.eleve@ecole-navale.fr
@Desc    :   Module to save, extract and display data from xarray and .nc file
"""

# ======================================================================================================================
# Import
# ======================================================================================================================

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy as sc
import xarray as xr
import cv2
import soundfile as sf

# ======================================================================================================================
# Functions
# ======================================================================================================================

def plot_spectro(
        spec: np.ndarray,
        freq: np.ndarray,
        time: np.ndarray,
        gain = 0,
        title = r'Spectrogram',
        size = [16,4],
        unit = 'min'
):
    """
    Display any full spectrogram or spectrogram of an isolated pattern

    Parameters
    ----------
    spec : ndarray
        Array of spectrogram to display.
    freq : ndarray
        Array of sample frequencies (Hertz).
    time : ndarray
        Array of sample time (Seconds, but displayed in minutes).
    gain : int, optional
        Display gain (dB). Default is 0.
    title : str, optional
        Title of the figure. Default is "Spectrogram".
    size : arraylike, optional
        Size of the outputed figure. Default is [16,4].
    unit : str
        'min' or 's', unit of the displayed spectrogram. Default is 'min'.

    Returns
    -------
    None

    """


    font = {'family' : 'sans-serif', 'size'   : 18}
    matplotlib.rc('font', **font)
    params = {"font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)
    matplotlib.rc('text', usetex=False)

    matplotlib.rcParams["mathtext.fontset"]='cm'
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    if unit == 'min':
        fig = plt.figure(figsize=size)
        im = plt.pcolormesh(time/60,freq,20*np.log10(abs(spec)))
        bmax = -gain
        bmin = bmax - 80

        plt.xlabel(r'$\rm{Time\ [min]}$')
        plt.ylabel(r'$f\ \rm{[Hz]}$')
        plt.title(title, loc= 'center')
    if unit == 's':
        fig = plt.figure(figsize=size)
        im = plt.pcolormesh(time,freq,20*np.log10(abs(spec)))
        bmax = -gain
        bmin = bmax - 80

        plt.xlabel(r'$\rm{Time\ [s]}$')
        plt.ylabel(r'$f\ \rm{[Hz]}$')
        plt.title(title, loc= 'center')

    plt.clim([bmin,bmax])
    plt.colorbar(im, label= '[dB]')
    fig.set_facecolor("white")
    plt.show()
    return

#=======================================================================================================================

def build_ds(
        full_spec: np.ndarray,
        freq: np.ndarray,
        time: np.ndarray,
        label_path: str,
) -> xr.Dataset :
    """
    Build an xarray dataset to save the individual patterns

    Parameters
    ----------
    full_spec : ndarray
        The full spectrogram which contains the patterns.
    freq : ndarray
        Array of sample frequencies (Hertz).
    time : ndarray
        Array of sample time (Seconds).
    label_path : str
        Path the the label file containing the pattern data

    Returns
    -------
    ds : xr.Dataset
        Fully built dataset containing the original spectrogram as well as individual patterns.

    """
        
    with open(label_path) as file:
        contents = file.read()
    line_count = contents.count('\n')//2
    list_content = contents.split('\n')

    ds = xr.Dataset(
    data_vars=dict(
            spectrogram_real = (["freq", "time"], full_spec.real),
            spectrogram_imag = (["freq", "time"], full_spec.imag),
    ),
    coords=dict(
            index = range(line_count),
            freq = freq,
            time = time,
    ),
    )

    mask = np.stack([np.full_like(full_spec, np.nan)]*line_count, axis=0)
    mask_f = np.stack([np.full_like(freq, np.nan)]*line_count, axis=0)
    mask_t = np.stack([np.full_like(time, np.nan)]*line_count, axis=0)
    pattern = []
    for i in range(line_count):
        delta_time = list_content[2*i].split()
        delta_freq = list_content[2*i+1][3:].split()

        pattern.append(delta_time[2])
        t1, t2 = float(delta_time[0]), float(delta_time[1])
        tmoy, dt = (t1+t2)/2, 7
        tmin = tmoy-dt/2 if tmoy-dt/2 > 0 else 0
        tmax = tmoy+dt/2 if tmoy+dt/2 < 300 else 300
        fmin, fmax = 0, 250

        fmask = (freq >= fmin) & (freq <= fmax)
        tmask = (time >= tmin) & (time <= tmax)

        fslice = np.full_like(freq, np.nan)
        tslice = np.full_like(time, np.nan)
        fslice[np.ix_(fmask)] = np.copy(freq[np.ix_(fmask)])
        tslice[np.ix_(tmask)] = np.copy(time[np.ix_(tmask)])
        mask_f[i] = np.copy(fslice)
        mask_t[i] = np.copy(tslice)

        spec_data = np.full_like(full_spec, np.nan)
        spec_data[np.ix_(fmask, tmask)] = np.copy(full_spec[np.ix_(fmask, tmask)])
        mask[i] = np.copy(spec_data)


    ds = ds.assign(mask_real = (["index", "freq", "time"],mask.real))
    ds = ds.assign(mask_imag = (["index", "freq", "time"],mask.imag))
    ds = ds.assign(mask_f = (["index", "freq"], mask_f))
    ds = ds.assign(mask_t = (["index", "time"], mask_t))
    ds = ds.assign(pattern = (["index"], pattern))

    return ds
    
#=======================================================================================================================

def open_ds(
        path: str,
        index: int,
        display: bool = False
) -> tuple :
    """
    Open a NetCDF (.nc) file and extract the xarray dataset and information about a singular index.

    Parameters
    ----------
    path : str
        String path to the file to open.
    index : int
        Index of the wanted pattern.
    display : bool
        If True, the function will output the spectrogram of the pattern. Default is false.

    Returns
    -------
    freq : ndarray
        Array of sample frequencies (Hertz).
    time : ndarray
        Array of sample time (Seconds).
    spec : ndarray
        The spectrogram of the extracted pattern.
    p : str
        The label name of the extracted pattern.
    ds : xr.Dataset
        The full dataset as it was saved.

    """
      
    ds = xr.open_dataset(path)
    spec = np.copy(ds.mask_real[index] + 1j*ds.mask_imag[index])

    freq = np.copy(ds.mask_f[index])
    freq = freq[~np.isnan(freq)]

    time = np.copy(ds.mask_t[index])
    time = time[~np.isnan(time)]

    spec = spec[~np.isnan(spec)].reshape(freq.size, time.size)
    p = ds.pattern[index].data

    if display:
            plot_spectro(spec= spec, freq= freq, time= time, gain=np.max(abs(spec))+30, title= path[-18:-3] + ' index : '+str(index)+' / pattern : '+p, size= [7,5], unit= 's')

    return freq, time, spec, p, ds

#=======================================================================================================================

def spec_ds(
        path: str,
        display: bool = False
) -> tuple :
    """
    Open a NetCDF (.nc) file and extract the xarray dataset and information about the full spectrogram.

    Parameters
    ----------
    path : str
        String path to the file to open.
    display : bool
        If True, the function will output the spectrogram of the dataset. Default is false.

    Returns
    -------
    freq : ndarray
        Array of sample frequencies (Hertz).
    time : ndarray
        Array of sample time (Seconds).
    spec : ndarray
        The full spectrogram which contains the patterns.
    n_pat : int
        The number of pattern in the dataset.
    ds : xarray dataset
        The full dataset as it was saved.

    """
    
    ds = xr.open_dataset(path)
    fmin, fmax = 0, 250
    fmask = (ds.freq >= fmin) & (ds.freq <= fmax)
    freq = np.copy(ds.freq[fmask])
    time = np.copy(ds.time)
    spec = np.copy(ds.spectrogram_real[fmask,:]+ 1j*ds.spectrogram_imag[fmask,:])
    n_pat = ds.index.size
    
    if display : 
        plot_spectro(spec= spec, freq= freq, time= time, gain= np.max(abs(spec))+30, title= path[-18:-3])
    
    return freq, time, spec, n_pat, ds

#=======================================================================================================================

def snr_test(
    spec: np.ndarray,
    limit: int = -70,
    threshold: int = 15000
) -> bool:
    """
    Computes the signal to noise ratio of given spectrogram and tests whether it is above a certain threshold.
    
    Parameters
    ----------
    
    spec : ndarray
        Array of the spectrogram to compute.
    limit : int
        Value above which the SNR will be tested.
    Threshold : int
        Threshold above which we consider the output to be positive.
        
    Returns
    -------
    
    out : bool
        Boolean which indicates if we consider to be a signal or not.
        
    """
    
    spec_aff = 20*np.log10(abs(spec))
    spec_filt = sc.ndimage.gaussian_filter(spec_aff,2)
    histogram_unfilt, bins = np.histogram(spec_aff, bins= 128, range= (-100,0))
    histogram_filt, _ = np.histogram(spec_filt, bins= 128, range= (-100,0))
    histogram = histogram_unfilt/(histogram_filt+10e-5)
    test = np.sum(histogram[bins[:-1]>limit])
    return test>=threshold

#=======================================================================================================================

def ssd_match(
    file_path: str,
    template: np.ndarray,
    score_threshold: int,
    nms_threshold: float
) -> tuple:
    """
    Searches for matches of the given template in the given audio file.
    
    Parameters
    ----------
    
    file_path : str
        Path to the .flac audio file.
    template : ndarray
        Template to search in the audio file. Should be a 0-255 greyscale image.
    score_threshold : int
        Score above which results will be considered as matches.
    nms_threshold : float
        Overlap between snapshots. Must be between 0 and 1.

    Returns
    -------
    
    picked_boxes : list
        List containing the boxes of the matching results.
    picked_scores : list
        Scores of those boxes.
    """
    
    audio, fs = sf.read(file_path)
    fmin, fmax = 0, 250
    
    n_fft = 4096
    hop_length = n_fft*0.2
    f, t, spectro = sc.signal.stft(audio,fs,'hann',nperseg=n_fft,noverlap=n_fft-hop_length)
    fmask = (f >= fmin) & (f <= fmax)
    tmask = (t >= 0)

    fslice = np.full_like(f, np.nan)
    fslice[np.ix_(fmask)] = np.copy(f[np.ix_(fmask)])
    fslice = fslice[~np.isnan(fslice)]

    spec_data = np.full_like(spectro, np.nan)
    spec_data[np.ix_(fmask, tmask)] = np.copy(spectro[np.ix_(fmask, tmask)])
    spec_data = spec_data[~np.isnan(spec_data)].reshape(fslice.size, t.size)
    
    spec_img = 20*np.log10(np.abs(spec_data))
    spec_img = (spec_img.astype(np.float32) - spec_img.mean()) / (spec_img.std() + 1e-5)
    spec_img = np.flipud(spec_img)
    spec_img[165:171] = spec_img[159:165]
    spec_img = cv2.normalize(spec_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec = (cv2.blur(spec_img,(3,3)))
    sobel_spec = cv2.Sobel(spec, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_spec = cv2.convertScaleAbs(sobel_spec)
    sobel_spec = (cv2.blur(sobel_spec*1.5,(3,3))).astype(np.uint8)
    spec[sobel_spec<110] = 0
    spec = (cv2.blur(spec,(9,3)))
    
    res = cv2.matchTemplate(spec, template, cv2.TM_SQDIFF) 
    
    h, w = template.shape

    indices = np.argsort(res.ravel())[:]
    ys, xs = np.unravel_index(indices, res.shape)
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y) in zip(xs, ys)]
    scores = [float(1e6/res[y, x]) for (x, y) in zip(xs, ys)]

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    picked_times = []
    
    if len(indices) != 0 :
        picked_boxes = [boxes[i] for i in indices]
        for (x, _, _, _) in picked_boxes:
            x_time = t[int(x+w/2)]
            picked_times.append(x_time)
        picked_scores = [scores[i] for i in indices]
        return picked_times, picked_scores
    else: 
        return [], []
