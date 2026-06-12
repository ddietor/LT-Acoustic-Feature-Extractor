# %% Libs:
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime as dt
import matplotlib.dates as md
import pandas as pd
import scipy as scp
import matplotlib.patches as patches

from matplotlib import mlab

# %% OctaveBandsCalculation function:
def OctaveBandsCalculation(fs,octave=1):
    '''
    Calculate the central frequencies of the first 30 bands for a specific octave.

    Extended description:
    This function computes the central frequencies of the octave bands up to the first 30 bands.
    The octave band value can be specified (e.g., 1, 1/3, etc.).

    Parameters:
    - fs (float): Sampling frequency [Hz].
    - octave (float, optional): Value of an octave band for the calculation. Default is 1.

    Returns:
    - OctaveBands (DataFrame): DataFrame with columns ['fc', 'fini', 'fend']:
                               - fc: Central frequency in the band [Hz].
                               - fini: Initial frequency of the band [Hz].
                               - fend: Final frequency of the band [Hz].

    Application:
    OctaveBands = OctaveBandsCalculation(fs, octave=1/3)
    
    Created/Last modified: 2024-04-05
    '''
    # Function implementation goes here    
    # fini = 12.401570718501562
    # fini = 9.843133202303695
    # fini = 7.812499999999999
    # fini = 6.200785359250778
    # fini = 4.921566601151848
    fini = 3.9062499999999996
    f_low = fini/np.sqrt(2**octave)
    f_up = fini*np.sqrt(2**octave)
    
    columns_name = ['fc','fini','fend']
    OctaveBands = []
    
    Nbands = 55
    fcs = (fini*((2**(octave))**(np.arange(0, Nbands))))
    for fc in fcs: 
        f_low = fc/np.sqrt(2**octave)
        f_up = fc*np.sqrt(2**octave)    
        if f_up <= fs/2:
            OctaveBands.append(dict(zip(columns_name,[fc,f_low,f_up])))
        else:
            break
    OctaveBands = pd.DataFrame(data=OctaveBands, columns = columns_name)
       
    return OctaveBands 

# %% spl_calculation function:
def spl_calculation(tspect,fspect,psd,OctaveBands):
    '''
    Calculate 1/n-octave band SPL from a PSD matrix.

    Summary:
    This function integrates a power spectral density (PSD) over predefined
    octave or 1/n-octave frequency bands to compute sound pressure levels (SPL)
    as a function of time. The integration is performed in the linear domain
    and converted back to dB. Bands with no valid frequency bins are removed.

    Parameters:
    - tspect (array-like): Time vector [s] or time bins associated with PSD.
    - fspect (array-like): Frequency vector [Hz] corresponding to PSD rows.
    - psd (2D numpy.ndarray): PSD matrix in dB, shape (n_frequencies, n_times).
    - OctaveBands (pandas.DataFrame): Table containing at least:
        * 'fini' (float): Lower frequency bound of each band [Hz]
        * 'fend' (float): Upper frequency bound of each band [Hz]

    Returns:
    - spl (numpy.ndarray): SPL per band and time, shape (n_valid_bands, n_times)

    Example:
    spl = spl_calculation(tspect=tspect,fspect=fspect,psd=psd,OctaveBands=OctaveBands)

    Notes:
    - Integration is performed assuming uniform frequency spacing.
    - PSD is converted from dB to linear scale before integration.
    - Bands with no matching frequency bins are removed from the output.
    - Output retains band ordering after filtering invalid rows.

    Created/Last modified: 2026-06-12
    '''
    spl = np.ones((len(OctaveBands),len(tspect)))*np.nan
    fbin = fspect[1]-fspect[0]
    row2del = []
    for j in range(len(OctaveBands)):
        row = OctaveBands.iloc[j]
        fini = row['fini']
        fend = row['fend']
        indxs = np.where((fspect >= fini) & (fspect <= fend))[0]
        if len(indxs)>1:
            spl[j,:] = 10*np.log10(np.sum(10**(psd[indxs,:]/10)*fbin, axis=0))
        elif len(indxs)==1: 
            spl[j,:] = 10*np.log10(10**(psd[indxs,:]/10)*fbin)
        elif len(indxs)==0:
            row2del.append(j)
    if len(row2del)>0:
        spl = np.delete(spl, row2del, axis=0)
        OctaveBands = OctaveBands.drop(row2del)
    return spl

# %% pltOctaveSPL function
def pltSPL(spl, tspect, OctaveBands, FileName='', SPLmin=None, SPLmax=None, FontSize=12, c_map='jet', path2save=None, filename=None):
    """
    Plot 1/n-octave SPL data as a time-frequency representation.

    Parameters:
    - spl (2D array): SPL values [dB re 1 uPa] (rows = octave bands, cols = time bins)
    - tspect (1D array): Time vector [s] or unix timestamps
    - OctaveBands (DataFrame): Octave band info with column 'fc' for central frequencies
    - FileName (str, optional): File or plot name
    - SPLmin, SPLmax (float, optional): Color scale limits
    - FontSize (int, optional): Font size for labels
    - c_map (str or Colormap, optional): Colormap
    - path2save (str, optional): Path to save figure
    - filename (str, optional): Name of file (without extension)

    Returns:
    - None
    """

    Tres = tspect[1] - tspect[0]
    y = np.arange(len(OctaveBands['fc'])+1) - 0.5
    yticks_str = [str(int(f//1)) for f in OctaveBands['fc']]

    # Color limits
    if SPLmin is None:
        SPLmin = np.nanmin(spl)
    if SPLmax is None:
        SPLmax = np.nanmax(spl)

    plt.figure(figsize=(14,8))
    taxis2plt = tspect
    # Plot
    plt.pcolormesh(taxis2plt, y[1:], spl, cmap=c_map, vmin=SPLmin, vmax=SPLmax)
    # Title
    if np.round(Tres*1e3) == 0:
        titl_str = r'Tres: %.2f $\mu$s' %(Tres*1e6)
    elif np.round(Tres) == 0:
        titl_str = r'Tres: %.2f ms' %(Tres*1e3)
    else:
        titl_str = r'Tres: %.2f s' %(Tres)
    plt.title(FileName+'\n'+titl_str, fontsize=FontSize+2)

    # Colorbar
    cbar = plt.colorbar()
    cbar.set_label(r'SPL [dB re 1$\mu$Pa]', rotation=270, verticalalignment='baseline', fontsize=FontSize)
    cbar.ax.tick_params(labelsize=FontSize)

    # Labels
    plt.ylabel('Frequency [Hz]', fontsize=FontSize)
    plt.xlabel('Time [s]', fontsize=FontSize)
    plt.yticks(y[1:], yticks_str)
    plt.tick_params(axis='both', which='major', labelsize=FontSize)
    plt.tick_params(axis='both', which='minor', labelsize=FontSize)
    plt.tight_layout()

    # Save
    if path2save is not None and filename is not None:
        os.makedirs(path2save, exist_ok=True)
        plt.savefig(os.path.join(path2save, filename+'.png'), bbox_inches='tight', dpi=150)

    plt.show()