# %% Libs:
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import mlab


# %% spectrogram_resolution function:
def spectrogram_resolution(NFFT, fs, overlap=0):
    """
    Computes spectrogram temporal and frequency resolution parameters.

    This function calculates three key quantities used in time-frequency analysis:
    the temporal resolution between FFT frames, the frequency-bin spacing, and the
    minimum resolvable (non-zero) frequency. These formulas follow standard DSP
    definitions and the formulations presented in:

    Diego-Tortosa, D.; et.al. Effective Strategies for Automatic Analysis of Acoustic
    Signals in Long-Term Monitoring. J. Mar. Sci. Eng. 2025, 13, 454.
    DOI: 10.3390/jmse13030454

    Parameters:
    - NFFT (int): Number of samples used for each FFT segment.
    - fs (float): Sampling frequency in Hz.
    - overlap (float, optional): Fractional overlap between consecutive FFT windows 
      (0 ≤ overlap < 1). Default is 0.

    Returns:
    - Tres (float): Temporal resolution between FFT frames, computed as:
          Tres = NFFT * (1 - overlap) / fs
    - Fres (float): Frequency-bin resolution, computed as:
          Fres = fs / NFFT
    - Fmin (float): Minimum resolvable non-zero frequency, computed as:
          Fmin = 2 * fs / NFFT

    Application:
    Tres, Fres, Fmin = spectrogram_resolution(NFFT=1024, fs=48000, overlap=0.5)

    Created/Last modified: 2025-12-10
    """
    Tres = NFFT*(1-overlap)/fs
    Fres = fs/NFFT
    Fmin = 2*fs/NFFT
    # info_spect = [NFFT,Tres,Fres,Fmin,overlap]
    return Tres, Fres, Fmin

# %% NFFTselector function:
def NFFTselector(fs,tbin_max,fbin_max,fvalid_min,over=0):
    '''
    Study the appropriate NFFT values based on specified parameters.

    Parameters:
    - fs (float): Sampling frequency [Hz].
    - tbin_max (float): Maximum time resolution [s].
    - fbin_max (float): Maximum frequency resolution [Hz].
    - fvalid_min (float): Minimum valid frequency [Hz].
    - over (float, optional): Overlap factor. Defaults to 0.

    Returns:
    - NFFTdf (pandas.DataFrame): DataFrame containing selected NFFT values and related parameters.

    Raises:
    - ValueError: If the over value is not between 0 and 1, or if there is an error in the input parameters.

    Application:
    NFFTdf = NFFTselector(fs, tbin_max, fbin_max, fvalid_min, over=1)

    Created/Last modified: 2024-04-05
    '''
    # Function implementation goes here
    if fbin_max > fvalid_min/4:
        fbin_max = fvalid_min/4
        print('NFFTselector() New fbin_max: %.1f Hz' %fbin_max)

    columns_name = ['NFFT','tbin_s','fbin_Hz','fmin_Hz']
    NFFTdf = []

    if over==1:
        NFFT_max = tbin_max*fs
    elif over>=0 and over <1:
        NFFT_max = tbin_max*fs/(1-over)
    else:
        raise ValueError('The over value should be between 0 and 1')
    NFFT_max_b2 = 2**(np.floor(np.log2(NFFT_max)))
    NFFT_min_fbin = fs/fbin_max
    NFFT_min_fvalid = 2*fs/fvalid_min
    NFFT_min = max(NFFT_min_fbin,NFFT_min_fvalid)
    NFFT_min_b2 = 2**(np.ceil(np.log2(NFFT_min)))

    NFFT = int(NFFT_min_b2)
    while NFFT<=NFFT_max_b2:
        tbin = NFFT*(1-over)/fs
        fbin = fs/NFFT
        fvalid = 2*fs/NFFT
        NFFTdf.append(dict(zip(columns_name,[NFFT,tbin,fbin,fvalid])))

        NFFT = NFFT*2
    NFFTdf = pd.DataFrame(data=NFFTdf, columns = columns_name)
    if len(NFFTdf)<1:
        raise ValueError('Error in the input parameters. Need to extend tbin_max (%.6f s), fbin_max (%.1f Hz), or fvalid_min(%.1f Hz). Otherwise see if the fs can be reduced. ' %(tbin_max,fbin_max,fvalid_min))
    return  NFFTdf

# %% Spectrogram_analysis_nfft function:
def spectrogram_analysis_nfft(signal,fs,NFFT,overlap,plotter=False):
    '''
    Returns a spectrogram analysis using the selected Number of samples in the Fast Fourier Transform (NFFT).

    Parameters:
    - signal (numpy.ndarray): Signal array in A values [a.u.].
    - fs (float): Frequency sampling of the signal [Hz].
    - NFFT (int): Samples to produce the FFT in a bin.
    - overlap (float): Overlap coefficient in bins of the spectrogram. It should be between 0 and 1.
    - plotter (bool, optional): Flag to represent the spectrogram result. If True, a plot will be generated. Defaults to False.

    Returns:
    - tspect (numpy.ndarray): Time array of the spectrogram [s].
    - fspect (numpy.ndarray): Frequency array of the spectrogram [Hz].
    - psd (numpy.ndarray): The Power Spectral Density (PSD) matrix [dB (re 1A^2/Hz)].
    - info_spect (list): List with some properties of the calculated spectrogram [Nfft, tbin, fbin, fvalid, overlap].
        Nfft (int): Samples of Non-equidistant Fast Fourier Transform (NFFT).
        tbin (float): Time resolution of the spectrogram (taking into account the overlap) [s].
        fbin (float): Frequency resolution of the spectrogram [Hz].
        fvalid (float): Frequency lower limit of spectrogram [Hz].
        overlap (float): Overlap coefficient in bins of the spectrogram.

    Application:
    tspect, fspect, psd, info_spect = spectrogram_analysis_nfft(signal, fs, Nfft, overlap, plotter=True)
    
    Created/Last modified: 2025-12-04
    * Now the info_spect has also the overlap info: [NFFT,tbin,fbin,fvalid,overlap]
    '''
    # Function implementation goes here
    Nover=int(np.floor(NFFT*overlap))
    fvalid = np.ceil(2*fs/NFFT)
    [pspect, fspect, tspect] = mlab.specgram(signal, NFFT = NFFT, Fs = fs, window = np.hamming(NFFT), noverlap = Nover, mode='psd')
    fbin = fspect[1]-fspect[0]
    tbin = tspect[1]-tspect[0]

    psd=10*np.log10(pspect) #PSD [dB re 1A^2/Hz]
    
    info_spect = [NFFT,tbin,fbin,fvalid,overlap]
    
    if plotter:      
        psd[np.isinf(psd)] = np.nan
        PSDmin = np.nanmin(psd[np.where((fspect >= fvalid))[0],:]) #dB
        PSDmax = np.nanmax(psd[np.where((fspect >= fvalid))[0],:]) #dB
        
        plt.figure()
        # getting the original colormap using cm.get_cmap() function
        orig_map=plt.colormaps.get_cmap('hot')        
        reversed_map = orig_map.reversed()
        if max(fspect)<=5e3:
            plt.pcolormesh(tspect, fspect, psd, cmap=reversed_map, vmin=PSDmin, vmax=PSDmax)
            plt.axhline(fvalid,color='black',linestyle='--',linewidth=4)
            plt.ylabel('Frequency [Hz]')
        else:
            plt.pcolormesh(tspect, fspect*1e-3, psd, cmap=reversed_map, vmin=PSDmin, vmax=PSDmax)
            plt.axhline(fvalid*1e-3,color='black',linestyle='--',linewidth=4)
            plt.ylabel('Frequency [kHz]')
        plt.title('spectrogram_analysis_nfft\n%i-NFFT (overlap: %i%% ; f$_{ok}\\geq$%i Hz)' %(NFFT,int(overlap*100),fvalid))
        cbar = plt.colorbar()
        cbar.set_label('PSD [dB re 1A$^2$/Hz]', rotation=270, verticalalignment='baseline')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()
    
    return tspect,fspect,psd,info_spect

# %% pltSpect function:
def pltSpect(psd,tspect,fspect,info_spect,PSDmin=None,PSDmax=None,title_str=None,CbarLabel=None,CbarColor=None,Xlims=None,Ylims=None, path2save=None, filename=None):
    """
    Plot a time-frequency spectrogram with PSD color scaling.

    This function generates a spectrogram plot using a power spectral density
    (PSD) matrix and associated time and frequency vectors. It supports automatic
    scaling of color limits, frequency unit conversion (Hz / kHz), and optional
    customization of plot appearance such as colorbar label, colormap, axis
    limits, title, and saving to disk.

    Parameters:
    - psd (2D array-like): Power Spectral Density matrix [dB re 1 A^2/Hz].
    - tspect (1D array-like): Time vector corresponding to spectrogram columns [s].
    - fspect (1D array-like): Frequency vector corresponding to spectrogram rows [Hz].
    - info_spect (tuple): Spectrogram parameters
        (NFFT, tbin, fbin, fvalid, overlap).
        - NFFT (int): FFT length.
        - tbin (float): Time bin size [s].
        - fbin (float): Frequency bin size [Hz].
        - fvalid (float): Minimum valid frequency threshold [Hz].
        - overlap (float): FFT overlap fraction (0–1).
    - PSDmin (float or None): Minimum PSD value for color scaling [dB].
    - PSDmax (float or None): Maximum PSD value for color scaling [dB].
    - title_str (str, optional): Title of the spectrogram plot.
    - CbarLabel (str, optional): Label for the colorbar.
    - CbarColor (str, optional): Colormap name.
    - Xlims (tuple of float, optional): Time-axis limits (tmin, tmax) [s].
    - Ylims (tuple of float, optional): Frequency-axis limits (fmin, fmax) [Hz].
    - path2save (str, optional): Path to save figure.
    - filename (str, optional): Name of file (without extension).

    Returns:
    - None

    Application:
    pltSpect(psd, tspect, fspect, info_spect)

    Created/Last modified: 2026-01-13
    """
    
    NFFT,tbin,fbin,fvalid,overlap = info_spect 
    
    if Ylims is None:
        Ylims = (fspect[0]-fbin/2,fspect[-1]+fbin/2)
    if Xlims is None:
        Xlims = (tspect[0]-tbin/2,tspect[-1]+tbin/2)
    if CbarColor is None:
        CbarColor = 'hot_r'
    if CbarLabel is None:
        CbarLabel = 'PSD [dB re 1A$^2$/Hz]'
    if title_str is None:
        title_str = f'pltSpect()\n{NFFT}-NFFT (overlap: {int(overlap*100)}%; f$_{{ok}}\\geq${fvalid} Hz)'
    if PSDmin is None or PSDmin >= PSDmax:
        PSDmin = np.nanmin(psd)
    if PSDmax is None or PSDmax <= PSDmin:
        PSDmax = np.nanmax(psd)
        
    plt.figure(figsize=(14, 8))
    # getting the original colormap using cm.get_cmap() function
    orig_map=plt.colormaps.get_cmap(CbarColor)        
    if max(fspect)<=5e3:
        plt.pcolormesh(tspect, fspect, psd, cmap=orig_map, vmin=PSDmin, vmax=PSDmax)
        plt.axhline(fvalid,color='black',linestyle='--',linewidth=4)
        plt.ylabel('Frequency [Hz]')
        plt.ylim(Ylims)
    else:
        plt.pcolormesh(tspect, fspect*1e-3, psd, cmap=orig_map, vmin=PSDmin, vmax=PSDmax)
        plt.axhline(fvalid*1e-3,color='black',linestyle='--',linewidth=4)
        plt.ylabel('Frequency [kHz]')
        plt.ylim((Ylims[0]*1e-3, Ylims[1]*1e-3))
    plt.title(title_str)
    plt.xlim(Xlims)
    cbar = plt.colorbar()
    cbar.set_label(CbarLabel, rotation=270, verticalalignment='baseline')
    plt.xlabel('Time [s]')
    plt.tight_layout()

    if path2save is not None and filename is not None:
        os.makedirs(path2save, exist_ok=True)
        plt.savefig(os.path.join(path2save, filename + ".png"),bbox_inches='tight', dpi=150)

    plt.show()
