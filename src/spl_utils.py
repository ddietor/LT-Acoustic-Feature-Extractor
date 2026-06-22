# %% Libs:
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as md
import pandas as pd
import warnings


# %% OctaveBandCalculation function:
def OctaveBandCalculation(freq,octave=1):
    '''
    Calculate the central frequency and frequency bounds for the specified frequency within an octave band.

    Extended description:
    This function calculates the central frequency and frequency bounds (lower and upper) for a given frequency
    within an octave band. 

    Parameters:
    - freq (float): The frequency for which to calculate the octave band [Hz].
    - octave (float, optional): Value of the octave band. Default is 1.

    Returns:
    - fc (float): Central frequency of the octave band in which the specified frequency falls [Hz].
    - f_low (float): Lower bound of the octave band [Hz].
    - f_up (float): Upper bound of the octave band [Hz].

    Application:
    fc, f_low, f_up = OctaveBandCalculation(freq, octave=1/3)

    Created/Last modified: 2025-01-17
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
    
    if freq < fini:
        freq = fini
        warnings.warn("WARNING from OctaveBandCalculation(): freq < fini. Solution: freq=fini", UserWarning,stacklevel=2)
    
    Nbands = 55
    fcs = (fini*((2**(octave))**(np.arange(0, Nbands))))
    for fc in fcs: 
        f_low = fc/np.sqrt(2**octave)
        f_up = fc*np.sqrt(2**octave)    
        if freq >= f_low and freq <= f_up :
            break
       
    return fc,f_low,f_up 

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

# %% signalReaderSPLdata function:   
def signalReaderSPLdata(CSVfile, freq, octave_band, plotter=False):
    '''
    Reads a CSV file containing sound pressure level (SPL) data and extracts the SPL data corresponding to a given frequency.

    Parameters:
    - CSVfile (str): Path to the CSV file containing SPL data. The file should have time data in the first row and SPL data in subsequent rows.
    - freq (float): The frequency (Hz) for which the SPL data is to be extracted.
    - octave (float): Value of the octave band used in the CSVfile data. 
    - plotter (bool, optional): If True, plot the result. Default is False.

    Returns:
    - time_read (numpy.ndarray): Time array corresponding to the first row in the CSV file [s].
    - freq_read (float): The frequency (Hz) that corresponds to the given frequency within the specified octave band.
    - spl_read (numpy.ndarray): The SPL values corresponding to the frequency in `freq_read` [µPa].

    Application:
    time_read, freq_read, spl_read = signalReaderSPLdata(CSVfile, freq, octave_band)

    Example usage:
    time_read, freq_read, spl_read = signalReaderSPLdata(CSVfile, freq, octave_band)
    
    Created/Last modified: 2024-12-13
    '''
    # Read CSV data
    try:
        CSVdata = pd.read_csv(CSVfile,index_col=None,sep=';')
    except Exception as e:
        raise Exception(f"Error reading CSV file '{CSVfile}': {e}")
    
    # Extract time data (assumed to be in the second column onward)
    time_read = CSVdata.iloc[0, 1:].to_numpy()

    # Loop through frequencies and find matching band
    freq_read, spl_read = None, None  # Default values in case no match is found
    for indx, f in enumerate(CSVdata.iloc[1:, 0].to_numpy()):  # Assuming first column has the frequencies
        fc, fmin, fmax = OctaveBandCalculation(f, octave=octave_band)
        
        if freq >= fmin and freq <= fmax:
            freq_read = CSVdata.iloc[indx + 1, 0]  # Read the frequency value
            spl_read = CSVdata.iloc[indx + 1, 1:].to_numpy()  # SPL values for this frequency
            break  # Exit loop once we find the match
    
    # If no match is found, raise an exception
    if freq_read is None or spl_read is None:
        raise Exception(f"No data found for frequency {freq} Hz in the given octave band.")
    
    if plotter: 
        plt.figure()
        plt.xticks(rotation=15)
        ax=plt.gca()
        xfmt=md.DateFormatter('%Y-%m-%d %H:%M:%S.%f')
        ax.xaxis.set_major_formatter(xfmt)
        taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in time_read]
        plt.plot(taxis2plt, spl_read, color='blue')
        if octave_band == 1/3:
            plt.title(r'%s\nSPL$_{1/3 octave}$ %.1f Hz' %(os.path.basename(CSVfile),freq_read))
        else:
            plt.title(r'%s\nSPL$_{%f octave}$ %.1f Hz' %(os.path.basename(CSVfile),octave_band,freq_read))
        plt.ylabel(r'SPL [dB re 1$\mu$Pa]')
        plt.tight_layout()
        plt.show()
    
    
    return time_read, freq_read, spl_read