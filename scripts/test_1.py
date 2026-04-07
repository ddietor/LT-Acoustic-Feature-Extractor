# %% Libs:
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as md
import pandas as pd
import scipy as scp
from matplotlib import mlab
import warnings


script_name = 'test_1.py'
print(f"Executing: {script_name} ...")
# %% OvDE_BINreader function(): 
def OvDE_BINreader(BINfile,plotter=False):
    '''
    Reads the data from an OvDE bin file and returns the signal.

    Parameters:
    - BINfile (str): Path to the input BIN file.
    - plotter (bool, optional): Flag to plot the signal data. Default is False.

    Returns:
    - signal_t (numpy.ndarray): Time array of the signal [s].
    - signal (numpy.ndarray): Signal array in pressure units [µPa].
    - fs_onde (float): Sampling frequency of the signal [Hz].

    Application:
    signal_t, signal, fs_onde = OvDE_BINreader(BINfile, plotter=False)
    
    Created/Last modified: 2024-04-05
    '''
    def OvDEresample_acou(sample, tgps_secs, tgps_25ns, fs):
        '''
        Resamples OvDE acoustic data.

        Parameters:
        - sample (list): Input sample data.
        - tgps_secs (list): Seconds component of the GPS time.
        - tgps_25ns (list): 25ns component of the GPS time.
        - fs (float): Sampling frequency.

        Returns:
        - Resampled sample data.
        
        Created/Last modified: 2024-04-05
        '''
        # Function implementation goes here
        m=[]
        sample_new=[0]*300*fs
        time=[]
        for i in range(0,len(tgps_secs)):
            time.append(tgps_secs[i]+tgps_25ns[i]*25e-9)
            
        for i in range (0,len(time)):
            m.append(round((time[i]-time[0])/(192/(fs))))
            for j in range(0,192):
                sample_new[round((time[i]-time[0])/(192/(fs)))*192+j]=sample[i*192+j]
        return sample_new
    
    def OvDEread_iadp(filename):
        '''
        Reads IADP data from a OvDE binary file.

        Parameters:
        - filename (str): Path to the input binary file.

        Returns:
        - Tuple containing the data read from the file.
        
        Created/Last modified: 2024-04-05
        '''
        # Function implementation goes here
        ido=[]
        ids=[]
        idh=[]
        fs=[]
        tgps_year=[]
        tgps_days=[]
        tgps_secs=[]
        tgps_25ns=[]
        tgps_year_os=[]
        tgps_days_os=[]
        tgps_secs_os=[]
        tgps_25ns_os=[]
        nfrm=[]
        nbits=[]
        sample=[]
       
        IADP_PACKET_SIZE_BYTES = 192*4 + 28
        fp = open(filename, 'rb')
      
        while True:
            rdbuf = fp.read(IADP_PACKET_SIZE_BYTES)
            if len(rdbuf) < IADP_PACKET_SIZE_BYTES:
                break
           
            ido.append(rdbuf[0]);
            ids.append(rdbuf[1]);
            idh.append(rdbuf[2]);
            fs.append(rdbuf[3]);
        
            tgps_year.append(rdbuf[4])
            tgps_days.append(((rdbuf[5]) << 8) | (rdbuf[6]))
            tgps_secs.append(((rdbuf[7]) << 16) | ((rdbuf[8] ) << 8) | (rdbuf[9]))
            tgps_25ns.append(((rdbuf[10]) << 24) | ((rdbuf[11]) << 16) | (( rdbuf[12] ) << 8) | (rdbuf[13]))
            
            nfrm.append(rdbuf[14])
            nbits.append(rdbuf[15])
        
            tgps_year_os.append(rdbuf[16])
            tgps_days_os.append (((rdbuf[17]) << 8)  | (rdbuf[18]))
            tgps_secs_os.append (((rdbuf[19]) << 16) | ((rdbuf[20] ) << 8) | (rdbuf[21]))
            tgps_25ns_os.append (((rdbuf[22]) << 24) | ((rdbuf[23]) << 16) | (( rdbuf[24] ) << 8) | (rdbuf[25]))
    #        try:
    #            pippo=abs(((tgps_secs[-1]+tgps_25ns[-1]*25e-9)-(tgps_secs[-2]+tgps_25ns[-2]*25e-9))%3600)
    #            print(pippo)
    #        except:
    #            print('primo giro')
            for i in range(7,  192+7):
                sample0 = ((rdbuf[4*i+3]) << 24) |  ((rdbuf[(4*i)+2]) << 16) | ( ( rdbuf[(4*i)+1] ) << 8) | (rdbuf[(4*i)])
                if sample0 >  2147483647:
                    sample.append(sample0 - (2*(2147483647 + 1))) 
                else:
                    sample.append(sample0)  
            
        return ido,ids,idh,fs,tgps_year,tgps_days,tgps_secs,tgps_25ns,nbits,sample  
    
    # Function implementation goes here
    str_data =  os.path.basename(BINfile)[:-4].split('_')
    year = 2000+int(str_data[4])
    month = int(str_data[5])
    day = int(str_data[6])
    hour = int(str_data[7][:-1])
    minute = int(str_data[8])
    unixtime_ini = dt.datetime(year, month, day, hour, minute).timestamp()
    
    ido,ids,idh,fs,tgps_year,tgps_days,tgps_secs,tgps_25ns,nbits,sample = OvDEread_iadp(BINfile)
    fs_onde = fs[0]*1000
    sample_new=OvDEresample_acou(sample,tgps_secs,tgps_25ns,fs_onde)
    
    # signal = np.array(sample_new)
    Nbits = 32  # ACDC
    # Nbits = np.array(sample_new).dtype.itemsize*8 
    # Nbits = nbits[0] # 24!
    if Nbits != 32:
        raise ValueError(f"ERROR in Nbits ({Nbits}) of this BINfile") # Just to check the bits lecture!
    RVR = -172  # [dB re 1uPa/V]
    signal_uPa = np.array(sample_new)*((1/2**(Nbits-1))*2*np.sqrt(2)/10**(RVR/20)) # bins (samples) to pressure (uPa): OK
    signal = signal_uPa
    # signal = sample_new
    signal_t = np.arange(0, len(signal) / fs_onde, 1 / fs_onde) + unixtime_ini
    
    if plotter: 
        # Representing the signal: 
        plt.figure()
        plt.xticks(rotation=15)
        ax=plt.gca()
        xfmt=md.DateFormatter('%Y-%m-%d %H:%M:%S.%f')
        ax.xaxis.set_major_formatter(xfmt)
        taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in signal_t]
        plt.plot(taxis2plt, signal, color='blue')
        plt.title(os.path.basename(BINfile).split('.')[0])
        plt.ylabel(r'Amplitude [$\mu$Pa]')
        plt.tight_layout()
        plt.show()
    
    return signal_t,signal,fs_onde

# %% Reading the data:
BinFile = r"C:\Users\Didac\Documents\ddietor\LT-Acoustic-Feature-Extractor\data_test\onde2_0_0_3_17_02_26_00h_15.bin" #OvDE-2: Sismic
signal_t, signal_uPa, fs = OvDE_BINreader(BinFile,plotter=False)
Tdur = signal_t[-1]-signal_t[0]
print(f"BinFile {os.path.basename(BinFile)} read successfully! \nDuration: {Tdur:.2f} s, fs: {fs*1e-3:.1f} kHz")
FileName = os.path.basename(BinFile).split('.')[0]

# %% Parameters:
datapath2save = r".\results_test"
if not os.path.exists(datapath2save):
    os.makedirs(datapath2save)
print(datapath2save,' is ready!')

# %% signal to wav:
factor_scale = 0.375*np.sqrt(2)
Nbits = 32 # ACDC
RVR_smo_onde_2 = -172 # [dB re 1uPa/V]
Pref = 1e-6 # [Pa]
Vref = 1 # [V]
FS_uPa = factor_scale * (Pref / Vref) * (1 / 10**(RVR_smo_onde_2/20)) * 1e6 #Full Scale in uPa
signal_norm = np.clip(signal_uPa / FS_uPa, -1, 1).astype(np.float32)
signal_bits = (signal_norm * (2**(Nbits-1))).astype(np.int32)
wavName = FileName+'.wav'
wavFile = os.path.join(datapath2save,wavName)
scp.io.wavfile.write(wavFile, int(fs), signal_bits)
print(f"{wavName} created with fs {fs*1e-3:.1f} kHz and {Nbits} bits")

print(np.max(np.abs(signal_uPa)))
# %% wav reader:
fs_read, signal_samples = scp.io.wavfile.read(wavFile)  
signal_norm = signal_samples.astype(np.float64) / 2**(Nbits-1)
signal_uPa_wav = signal_norm * FS_uPa
print(f"{wavName} readed!")
print(np.max(np.abs(signal_uPa_wav)))

# plt.figure()
# plt.xticks(rotation=15)
# ax=plt.gca()
# xfmt=md.DateFormatter('%Y-%m-%d %H:%M:%S.%f')
# ax.xaxis.set_major_formatter(xfmt)
# taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in signal_t]
# plt.plot(taxis2plt, signal_uPa_wav, color='black')
# plt.title(os.path.basename(wavFile).split('.')[0])
# plt.ylabel(r'Amplitude [$\mu$Pa]')
# plt.tight_layout()
# plt.show()

# %% Spectrogram parameters:
NFFT = 2048
overlap_bin = 0.5

Fbin = fs/NFFT
Tbin = (NFFT/fs)*(1-overlap_bin)
Fvalid = (2*fs/NFFT)
print(f"{NFFT}-NFFT and {overlap_bin*100:.0f}% overlap -> Fbin:{Fbin:.1f} Hz, Tbin:{Tbin:.3f} s, Fvalid:{Fvalid:.1f} Hz")

XlabelFormat,XlabelRot = '%H:%M:%S',20
FontSize = 14
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
        plt.title(r'spectrogram_analysis_nfft\n%i-NFFT (overlap: %i%% ; f$_{ok}\geq$%i Hz)' %(NFFT,int(overlap*100),fvalid))
        cbar = plt.colorbar()
        cbar.set_label('PSD [dB re 1A$^2$/Hz]', rotation=270, verticalalignment='baseline')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()
    
    return tspect,fspect,psd,info_spect

# %% Spectrogram HF calculation:
tspect,fspect,psd,spect_info = spectrogram_analysis_nfft(signal_uPa,fs,NFFT=NFFT,overlap=overlap_bin,plotter=False)
tspect = tspect + signal_t[0]
_,tbin,fbin,fvalid,_ = spect_info
 
PSD_high = np.column_stack([np.append(np.nan,fspect),np.vstack([tspect,psd])])
fspect_high = fspect.copy()

SpectName = FileName + '_PSDdata'

# %% Spectrogram HF representation:
PSDmin,PSDmax = 30,140 # [dB re 1uPa^2/Hz]
plt.ioff() # Turn interactive plotting off
plt.figure(figsize=(14,8))
plt.xticks(rotation=XlabelRot)
ax=plt.gca()
xfmt=md.DateFormatter(XlabelFormat)
ax.xaxis.set_major_formatter(xfmt)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in tspect]
c_map = plt.colormaps['jet']
# PSDmin,PSDmax = np.min(psd),np.max(psd)
Nvalid = np.where(fspect >= fvalid)[0][0]
fspect = fspect[Nvalid:]
psd = psd[Nvalid:,:]
if max(fspect)<=5e3:
    plt.pcolormesh(taxis2plt, fspect, psd, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
    # plt.axhline(fvalid,color='black',linestyle='--',linewidth=4)
    plt.ylabel('Frequency [Hz]', fontsize=FontSize)
    plt.ylim(fvalid,fspect[-1])
else:
    plt.pcolormesh(taxis2plt, fspect*1e-3, psd, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
    # plt.axhline(fvalid*1e-3,color='black',linestyle='--',linewidth=4)
    plt.ylabel('Frequency [kHz]', fontsize=FontSize)
    plt.ylim(fvalid*1e-3,fspect[-1]*1e-3)
plt.title(FileName)
# plt.title('%s\n%i-NFFT (overlap: %i%% ; f$_{ok}\geq$%i Hz)' %(FileName,NFFT,int(over*100),fvalid))
plt.title(r'%s\n%i-NFFT (overlap: %i%%)' %(FileName,NFFT,int(over*100)), fontsize=FontSize+2)
cbar = plt.colorbar()
cbar.set_label(r'PSD [dB re 1$\mu$Pa$^2$/Hz]', rotation=270, verticalalignment='baseline', fontsize=FontSize)
cbar.ax.tick_params(labelsize=FontSize)
# plt.xticks(color='white')
# ax.axes.xaxis.set_ticklabels([])
plt.xlabel('Time [HH:MM:SS]', fontsize=FontSize)
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()

# PSDdata png saving: 
save_PSDdata_png = True
if save_PSDdata_png:
    plt.savefig(os.path.join(datapath2save,SpectName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# %% Spectrogram HF saving (PSDdata):
save_PSDdata_csv = False
if save_PSDdata_csv:
    psd2csv = np.column_stack([np.append(np.nan,fspect),np.vstack([tspect,psd])])
    csvData = pd.DataFrame(psd2csv)
    csvData.to_csv(os.path.join(datapath2save,SpectName+'.csv'),index=False,header=False,sep =';')    

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

# %% 1/3 octave SPL-BL calculation (SPLdata):
OctaveBands_fs = OctaveBandsCalculation(fs,octave=1/3) 
OctaveBands = OctaveBands_fs[OctaveBands_fs['fini'] >= min(fspect[fspect>=fvalid])]
OctaveBands = OctaveBands.reset_index(drop=True)
spl = np.ones((len(OctaveBands),len(tspect)))*np.nan
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

SplName = FileName + '_SPLdata'

# %% 1/3 octave SPL-BL representation (SPLdata):
Tres = tspect[1]-tspect[0]
y = np.arange(len(OctaveBands['fc'])+1) - 0.5
yticks_str=[str(int(f//1)) for f in OctaveBands['fc']]
plt.ioff() # Turn interactive plotting off
plt.figure(figsize=(14,8))
plt.xticks(rotation=XlabelRot)
ax=plt.gca()
xfmt=md.DateFormatter(XlabelFormat)
ax.xaxis.set_major_formatter(xfmt)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in tspect]
plt.pcolormesh(taxis2plt, y[1:], spl, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
if np.round(Tres*1e3)==0:
    titl_str = r'Tres: %.2f $\mu$s' %(Tres*1e6)
elif np.round(Tres)==0:
    titl_str = r'Tres: %.2f ms' %(Tres*1e3)
else:
    titl_str = r'Tres: %.2f s' %(Tres)
plt.title(FileName+'\n'+titl_str, fontsize=FontSize+2)
cbar = plt.colorbar()
cbar.set_label(r'SPL [dB re 1$\mu$Pa]', rotation=270, verticalalignment='baseline', fontsize=FontSize)
cbar.ax.tick_params(labelsize=FontSize)
plt.ylabel('Frequency [Hz]', fontsize=FontSize)
plt.xlabel('Time [HH:MM:SS]', fontsize=FontSize)
plt.yticks(y[1:], yticks_str)
# plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show() 

# SPLdata png saving: 
save_SPLdata_png = True
if save_SPLdata_png:
    plt.savefig(os.path.join(datapath2save,SplName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# %% SPLdata csv saving: 
save_SPLdata_csv = True
if save_SPLdata_csv:
    spl2csv = np.column_stack([np.append(np.nan,OctaveBands['fc'].values),np.vstack([tspect,spl])])
    csvData = pd.DataFrame(spl2csv)
    csvData.to_csv(os.path.join(datapath2save,SplName+'.csv'),index=False,header=False,sep =';')    

# %% Spectrosum PSD calculation (PSDdata_cum):
PSDcumName = FileName + '_PSDdata_cum'
psd_mean = 10*np.log10(np.nanmean(10**(psd/10),axis=1))
psd_pctl25 = np.nanpercentile(psd,25,axis=1)
psd_pctl50 = np.nanpercentile(psd,50,axis=1)
psd_pctl75 = np.nanpercentile(psd,75,axis=1)
psd_pctl90 = np.nanpercentile(psd,90,axis=1)
psd_pctl95 = np.nanpercentile(psd,95,axis=1)
psd_pctl98 = np.nanpercentile(psd,98,axis=1)
psd_pctl99 = np.nanpercentile(psd,99,axis=1)

psd_mean_high = psd_mean.copy()  
# Create a DataFrame
psd_cum = pd.DataFrame({
    'Freq_Hz': fspect,
    'Mean': psd_mean,
    'PCTL25': psd_pctl25,
    'PCTL50': psd_pctl50,
    'PCTL75': psd_pctl75,
    'PCTL90': psd_pctl90,
    'PCTL95': psd_pctl95,
    'PCTL98': psd_pctl98,
    'PCTL99': psd_pctl99
})

# %% Spectrosum PSD representation (PSDdata_cum):
plt.ioff() # Turn interactive plotting off
fig = plt.figure(figsize=(14,8))
if max(fspect)<=5e3:
    plt.plot(fspect, psd_mean, color='gold', label='Mean')
    plt.plot(fspect, psd_pctl25, color='midnightblue', label='PCTL25th (%.1fs)' %(Tdur-Tdur*0.25))
    plt.plot(fspect, psd_pctl50, color='darkgreen', label='PCTL50th (%.1fs)' %(Tdur-Tdur*0.5))
    plt.plot(fspect, psd_pctl75, color='red', label='PCTL75th (%.1fs)' %(Tdur-Tdur*0.75))
    plt.plot(fspect, psd_pctl90, color='lime', label='PCTL90th (%.1fs)' %(Tdur-Tdur*0.9))
    plt.plot(fspect, psd_pctl95, color='aqua', label='PCTL95th (%.1fs)' %(Tdur-Tdur*0.95))
    plt.plot(fspect, psd_pctl98, color='fuchsia', label='PCTL98th (%.1fs)' %(Tdur-Tdur*0.98))
    plt.plot(fspect, psd_pctl99, color='lightpink', label='PCTL99th (%.1fs)' %(Tdur-Tdur*0.99))
    plt.xlim(fspect[0],fspect[-1])
    plt.xlabel('Frequency [Hz]', fontsize=FontSize)
else:
    plt.plot(fspect*1e-3, psd_mean, color='gold', label='Mean')
    plt.plot(fspect*1e-3, psd_pctl25, color='midnightblue', label='PCTL25th (%.1fs)' %(Tdur-Tdur*0.25))
    plt.plot(fspect*1e-3, psd_pctl50, color='darkgreen', label='PCTL50th (%.1fs)' %(Tdur-Tdur*0.5))
    plt.plot(fspect*1e-3, psd_pctl75, color='red', label='PCTL75th (%.1fs)' %(Tdur-Tdur*0.75))
    plt.plot(fspect*1e-3, psd_pctl90, color='lime', label='PCTL90th (%.1fs)' %(Tdur-Tdur*0.9))
    plt.plot(fspect*1e-3, psd_pctl95, color='aqua', label='PCTL95th (%.1fs)' %(Tdur-Tdur*0.95))
    plt.plot(fspect*1e-3, psd_pctl98, color='fuchsia', label='PCTL98th (%.1fs)' %(Tdur-Tdur*0.98))
    plt.plot(fspect*1e-3, psd_pctl99, color='lightpink', label='PCTL99th (%.1fs)' %(Tdur-Tdur*0.99))
    plt.xlim(fspect[0]*1e-3,fspect[-1]*1e-3)
    plt.xlabel('Frequency [kHz]', fontsize=FontSize)
plt.legend(ncol=4, loc='upper center', fontsize=FontSize-2)
plt.ylabel(r"PSD [dB re 1$\mu$Pa$^2$/Hz]", fontsize=FontSize)
plt.ylim(PSDmin,PSDmax) 
plt.grid(visible='on')
# plt.xscale('log')
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()  

# PSDdata_cum png saving: 
save_PSDdataCum_png = True
if save_PSDdataCum_png:
    plt.savefig(os.path.join(datapath2save,PSDcumName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# %% Spectrosum PSD saving (PSDdata_cum):
save_PSDdataCum_csv = True
if save_PSDdataCum_csv:
    psd_cum.to_csv(os.path.join(datapath2save,PSDcumName+'.csv'),index=False,header=True,sep =';')  

# %% Spectrosum SPL (SPLdata_cum) calculation:
SPLcumName = FileName + '_SPLdata_cum'
spl_mean = np.nanmean(spl,axis=1)
spl_pctl25 = np.nanpercentile(spl,25,axis=1)
spl_pctl50 = np.nanpercentile(spl,50,axis=1)
spl_pctl75 = np.nanpercentile(spl,75,axis=1)
spl_pctl90 = np.nanpercentile(spl,90,axis=1)
spl_pctl95 = np.nanpercentile(spl,95,axis=1)
spl_pctl98 = np.nanpercentile(spl,98,axis=1)
spl_pctl99 = np.nanpercentile(spl,99,axis=1)

# Create a DataFrame
spl_cum = pd.DataFrame({
    'Freq_Hz': OctaveBands['fc'].values,
    'Mean': spl_mean,
    'PCTL25': spl_pctl25,
    'PCTL50': spl_pctl50,
    'PCTL75': spl_pctl75,
    'PCTL90': spl_pctl90,
    'PCTL95': spl_pctl95,
    'PCTL98': spl_pctl98,
    'PCTL99': spl_pctl99
})

# %% Spectrosum SPL (SPLdata_cum) representation:
x = np.arange(len(OctaveBands['fc'])+1) 
barwidth = 1/9 #to adjust barwidth
xticks_str=[str(int(f//1)) for f in OctaveBands['fc']]
plt.ioff() # Turn interactive plotting off
fig = plt.figure(figsize=(8,8))
plt.bar(x[1:]-barwidth*5, spl_mean, color='gold', width=barwidth, align='edge', label='Mean')
plt.bar(x[1:]-barwidth*4, spl_pctl25, color='midnightblue', width=barwidth, align='edge', label='PCTL25th (%.1fs)' %(Tdur-Tdur*0.25))
plt.bar(x[1:]-barwidth*3, spl_pctl50, color='darkgreen', width=barwidth, align='edge', label='PCTL50th (%.1fs)' %(Tdur-Tdur*0.5))
plt.bar(x[1:]-barwidth*2, spl_pctl75, color='red', width=barwidth, align='edge', label='PCTL75th (%.1fs)' %(Tdur-Tdur*0.75))
plt.bar(x[1:]-barwidth, spl_pctl90, color='lime', width=barwidth, align='edge', label='PCTL90th (%.1fs)' %(Tdur-Tdur*0.9))
plt.bar(x[1:], spl_pctl95, color='aqua', width=barwidth, align='edge', label='PCTL95th (%.1fs)' %(Tdur-Tdur*0.95))
plt.bar(x[1:]+barwidth, spl_pctl98, color='fuchsia', width=barwidth, align='edge', label='PCTL98th (%.1fs)' %(Tdur-Tdur*0.98))
plt.bar(x[1:]+barwidth*2, spl_pctl99, color='lightpink', width=barwidth, align='edge', label='PCTL99th (%.1fs)' %(Tdur-Tdur*0.99))
plt.legend(ncol=2,loc='upper center',fontsize=FontSize-2)
plt.xticks(x[1:], xticks_str)
plt.xlabel(r'Frequency [Hz]', fontsize=FontSize)
plt.xticks(rotation=90)
plt.ylabel(r"SPL [dB re 1$\mu$Pa]", fontsize=FontSize)
plt.ylim(PSDmin,PSDmax)
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()       

# SPLdata_cum png saving: 
save_SPLdataCum_png = True
if save_SPLdataCum_png:
    plt.savefig(os.path.join(datapath2save,SPLcumName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# %% SPLdata_cum csv saving: 
save_SPLdataCum_csv = True
if save_SPLdataCum_csv:
    spl_cum.to_csv(os.path.join(datapath2save,SPLcumName+'.csv'),index=False,header=True,sep =';')  
    

# %% fs_factor function(): 
def fs_factor(fs, fmax_sel):
    '''
    Calculates the down-sampling factor to meet the maximum frequency constraint (fmax_sel).

    Parameters:
    - fs (float): Original sampling frequency [Hz].
    - fmax_sel (float): Maximum selected frequency [Hz].

    Returns:
    - factor (int): Down-sampling factor to achieve a maximum frequency below fmax_sel.
    - fs_new (float): Adjusted sampling frequency after down-sampling [Hz].

    Application:
    factor, fs_new = fs_factor(fs, fmax_sel)
    
    Created/Last modified: 2024-04-05
    '''
    # Function implementation goes here
    fs_new = fs
    factor = 1
    while fs_new / 2 >= fmax_sel:
        factor += 1
        fs_new = fs / factor
    if fs_new / 2 < fmax_sel:
        factor -= 1
        fs_new = fs / factor

    return factor, fs_new

# %% Spectrogram LF:
fmax_overlap = 1000 # [Hz]
factor_resamp,fs_resamp_expect = fs_factor(fs, fmax_overlap)
f_antialiasign = (fs_resamp_expect/2)*0.8 # [Hz]

# %% Spectrogram LF calculation (low_PSDdata):
signal_resamp = scp.signal.decimate(signal_uPa, factor_resamp)
fs_resamp = fs/factor_resamp
tspect,fspect,psd,spect_info = spectrogram_analysis_nfft(signal_resamp,fs_resamp,NFFT=NFFT,overlap=overlap_bin,plotter=0)
tspect = tspect + signal_t[0]
_,tbin,fbin,fvalid,_ = spect_info

PSD_low = np.column_stack([np.append(np.nan,fspect),np.vstack([tspect,psd])])

# %% Spectrogram LF representation (low_PSDdata):
SpectName = FileName + '_low_PSDdata'
plt.ioff() # Turn interactive plotting off
plt.figure(figsize=(14,8))
plt.xticks(rotation=XlabelRot)
ax=plt.gca()
xfmt=md.DateFormatter(XlabelFormat)
ax.xaxis.set_major_formatter(xfmt)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in tspect]
c_map = plt.colormaps['jet']
# PSDmin,PSDmax = np.min(psd),np.max(psd)
Nvalid = np.where(fspect >= fvalid)[0][0]
fspect = fspect[Nvalid:]
psd = psd[Nvalid:,:]
if max(fspect)<=5e3:
    plt.pcolormesh(taxis2plt, fspect, psd, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
    plt.axhline(f_antialiasign,color='black',linestyle='--',linewidth=4)
    plt.ylabel(r'Frequency [Hz]', fontsize=FontSize)
    plt.ylim(fspect[0],fspect[-1])
else:
    plt.pcolormesh(taxis2plt, fspect*1e-3, psd, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
    plt.axhline(f_antialiasign*1e-3,color='black',linestyle='--',linewidth=4)
    plt.ylabel(r'Frequency [kHz]', fontsize=FontSize)
    plt.ylim(fspect[0]*1e-3,fspect[-1]*1e-3)
plt.title(FileName)
plt.title(r'%s\n%i-NFFT (overlap: %i%%)' %(FileName,NFFT,int(overlap_bin*100)), fontsize=FontSize+2)
cbar = plt.colorbar()
cbar.set_label(r'PSD [dB re 1$\mu$Pa$^2$/Hz]', rotation=270, verticalalignment='baseline', fontsize=FontSize)
cbar.ax.tick_params(labelsize=FontSize)
# ax.axes.xaxis.set_ticklabels([])
plt.xlabel('Time [HH:MM:SS]', fontsize=FontSize)
# plt.yscale('log')
# if max(fspect)<=5e3:
#     yticks = [10, 50, 100, 500, 1000]  # Adjust as necessary
#     plt.yticks(yticks, [f'{int(tick)}' for tick in yticks])
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()

# PSDdata png saving: 
save_low_PSDdata_png = True
if save_low_PSDdata_png:
    plt.savefig(os.path.join(datapath2save,SpectName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# %% PSDdata csv saving: 
save_low_PSDdata_csv = True
if save_low_PSDdata_csv:
    psd2csv = np.column_stack([np.append(np.nan,fspect),np.vstack([tspect,psd])])
    csvData = pd.DataFrame(psd2csv)
    csvData.to_csv(os.path.join(datapath2save,SpectName+'.csv'),index=False,header=False,sep =';')    

# %% Low 1/3 octave SPL-BL:
OctaveBands_fs = OctaveBandsCalculation(fs_resamp,octave=1/3) 
OctaveBands = OctaveBands_fs[OctaveBands_fs['fini'] >= min(fspect[fspect>=fvalid])]
OctaveBands = OctaveBands[(OctaveBands['fini'] < f_antialiasign) & 
                          (OctaveBands['fend'] < f_antialiasign) & 
                          (OctaveBands['fc'] < f_antialiasign)]
OctaveBands = OctaveBands.reset_index(drop=True)
spl = np.ones((len(OctaveBands),len(tspect)))*np.nan
row2del = []
for j in range(len(OctaveBands)):
    row = OctaveBands.iloc[j]
    fini = row['fini']
    fend = row['fend']
    indxs = np.where((fspect >= fini) & (fspect <= fend))[0]
    if len(indxs)>1:
        # spl[j,:] = 10*np.log10(np.sum(10**(psd[indxs,:]/10), axis=0)*np.sqrt(fspect[indxs[-1]]-fspect[indxs[0]])/len(indxs)) #[dB re A]
        spl[j,:] = 10*np.log10(np.sum(10**(psd[indxs,:]/10)*fbin, axis=0)) #[dB re A]        
    elif len(indxs)==1: 
        # spl[j,:] = 10*np.log10(10**(psd[indxs,:]/10)*np.sqrt(1)/len(indxs)) #[dB re A]
        spl[j,:] = 10*np.log10(10**(psd[indxs,:]/10)*fbin) #[dB re A]
    elif len(indxs)==0:
        row2del.append(j)
if len(row2del)>0:
    spl = np.delete(spl, row2del, axis=0)
    OctaveBands = OctaveBands.drop(row2del)

SplName = FileName + '_low_SPLdata'
Tres = tspect[1]-tspect[0]
y = np.arange(len(OctaveBands['fc'])+1) - 0.5
yticks_str=[str(int(f//1)) for f in OctaveBands['fc']]
plt.ioff() # Turn interactive plotting off
plt.figure(figsize=(14,8))
plt.xticks(rotation=XlabelRot)
ax=plt.gca()
xfmt=md.DateFormatter(XlabelFormat)
ax.xaxis.set_major_formatter(xfmt)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in tspect]
plt.pcolormesh(taxis2plt, y[1:], spl, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
if np.round(Tres*1e3)==0:
    titl_str = r'Tres: %.2f $\mu$s' %(Tres*1e6)
elif np.round(Tres)==0:
    titl_str = r'Tres: %.2f ms' %(Tres*1e3)
else:
    titl_str = 'Tres: %.2f s' %(Tres)
plt.title(FileName+'\n'+titl_str, fontsize=FontSize+2)
cbar = plt.colorbar()
cbar.set_label(r'SPL [dB re 1$\mu$Pa]', rotation=270, verticalalignment='baseline', fontsize=FontSize)
cbar.ax.tick_params(labelsize=FontSize)
plt.ylabel('Frequency [Hz]', fontsize=FontSize)
plt.xlabel('Time [HH:MM:SS]', fontsize=FontSize)
plt.yticks(y[1:], yticks_str)
plt.xticks(rotation=40)
# plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show() 

# SPLdata png saving: 
save_low_SPLdata_png = True
if save_low_SPLdata_png:
    plt.savefig(os.path.join(datapath2save,SplName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# SPLdata csv saving: 
save_low_SPLdata_csv = True
if save_low_SPLdata_csv:
    spl2csv = np.column_stack([np.append(np.nan,OctaveBands['fc'].values),np.vstack([tspect,spl])])
    csvData = pd.DataFrame(spl2csv)
    csvData.to_csv(os.path.join(datapath2save,SplName+'.csv'),index=False,header=False,sep =';')    


# %% (3b) Low Spectrosum PSD: 
PSDcumName = FileName + '_low_PSDdata_cum'
psd_mean = 10*np.log10(np.nanmean(10**(psd/10),axis=1))
psd_pctl25 = np.nanpercentile(psd,25,axis=1)
psd_pctl50 = np.nanpercentile(psd,50,axis=1)
psd_pctl75 = np.nanpercentile(psd,75,axis=1)
psd_pctl90 = np.nanpercentile(psd,90,axis=1)
psd_pctl95 = np.nanpercentile(psd,95,axis=1)
psd_pctl98 = np.nanpercentile(psd,98,axis=1)
psd_pctl99 = np.nanpercentile(psd,99,axis=1)

# Create a DataFrame
psd_cum = pd.DataFrame({
    'Freq_Hz': fspect,
    'Mean': psd_mean,
    'PCTL25': psd_pctl25,
    'PCTL50': psd_pctl50,
    'PCTL75': psd_pctl75,
    'PCTL90': psd_pctl90,
    'PCTL95': psd_pctl95,
    'PCTL98': psd_pctl98,
    'PCTL99': psd_pctl99
})

plt.ioff() # Turn interactive plotting off
fig = plt.figure(figsize=(14,8))
if max(fspect)<=5e3:
    plt.plot(fspect, psd_mean, color='gold', label='Mean')
    plt.plot(fspect, psd_pctl25, color='midnightblue', label='PCTL25th (%.1fs)' %(Tdur-Tdur*0.25))
    plt.plot(fspect, psd_pctl50, color='darkgreen', label='PCTL50th (%.1fs)' %(Tdur-Tdur*0.5))
    plt.plot(fspect, psd_pctl75, color='red', label='PCTL75th (%.1fs)' %(Tdur-Tdur*0.75))
    plt.plot(fspect, psd_pctl90, color='lime', label='PCTL90th (%.1fs)' %(Tdur-Tdur*0.9))
    plt.plot(fspect, psd_pctl95, color='aqua', label='PCTL95th (%.1fs)' %(Tdur-Tdur*0.95))
    plt.plot(fspect, psd_pctl98, color='fuchsia', label='PCTL98th (%.1fs)' %(Tdur-Tdur*0.98))
    plt.plot(fspect, psd_pctl99, color='lightpink', label='PCTL99th (%.1fs)' %(Tdur-Tdur*0.99))
    plt.xlim(fspect[0],fspect[-1])
    plt.axvline(f_antialiasign,color='black',linestyle='--',linewidth=3)
    plt.xlabel('Frequency [Hz]', fontsize=FontSize)
else:
    plt.plot(fspect*1e-3, psd_mean, color='gold', label='Mean')
    plt.plot(fspect*1e-3, psd_pctl25, color='midnightblue', label='PCTL25th (%.1fs)' %(Tdur-Tdur*0.25))
    plt.plot(fspect*1e-3, psd_pctl50, color='darkgreen', label='PCTL50th (%.1fs)' %(Tdur-Tdur*0.5))
    plt.plot(fspect*1e-3, psd_pctl75, color='red', label='PCTL75th (%.1fs)' %(Tdur-Tdur*0.75))
    plt.plot(fspect*1e-3, psd_pctl90, color='lime', label='PCTL90th (%.1fs)' %(Tdur-Tdur*0.9))
    plt.plot(fspect*1e-3, psd_pctl95, color='aqua', label='PCTL95th (%.1fs)' %(Tdur-Tdur*0.95))
    plt.plot(fspect*1e-3, psd_pctl98, color='fuchsia', label='PCTL98th (%.1fs)' %(Tdur-Tdur*0.98))
    plt.plot(fspect*1e-3, psd_pctl99, color='lightpink', label='PCTL99th (%.1fs)' %(Tdur-Tdur*0.99))
    plt.xlim(fspect[0]*1e-3,fspect[-1]*1e-3)
    plt.axvline(f_antialiasign*1e-3,color='black',linestyle='--',linewidth=3)
    plt.xlabel('Frequency [kHz]', fontsize=FontSize)
plt.legend(ncol=4, loc='upper center', fontsize=FontSize-2)
plt.ylabel(r"PSD [dB re 1$\mu$Pa$^2$/Hz]", fontsize=FontSize)
plt.ylim(PSDmin,PSDmax)
# plt.xlim(0,f_antialiasign) 
plt.grid(visible='on')
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()  

# PSDdata_cum png saving: 
save_low_PSDdataCum_png = True
if save_low_PSDdataCum_png:
    plt.savefig(os.path.join(datapath2save,PSDcumName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# PSDdata_cum csv saving: 
save_low_PSDdataCum_csv = True
if save_low_PSDdataCum_csv:
    psd_cum.to_csv(os.path.join(datapath2save,PSDcumName+'.csv'),index=False,header=True,sep =';')  

# %% Low Spectrosum SPL:
SPLcumName = FileName + '_low_SPLdata_cum'
spl_mean = np.nanmean(spl,axis=1)
spl_pctl25 = np.nanpercentile(spl,25,axis=1)
spl_pctl50 = np.nanpercentile(spl,50,axis=1)
spl_pctl75 = np.nanpercentile(spl,75,axis=1)
spl_pctl90 = np.nanpercentile(spl,90,axis=1)
spl_pctl95 = np.nanpercentile(spl,95,axis=1)
spl_pctl98 = np.nanpercentile(spl,98,axis=1)
spl_pctl99 = np.nanpercentile(spl,99,axis=1)

# Create a DataFrame
spl_cum = pd.DataFrame({
    'Freq_Hz': OctaveBands['fc'].values,
    'Mean': spl_mean,
    'PCTL25': spl_pctl25,
    'PCTL50': spl_pctl50,
    'PCTL75': spl_pctl75,
    'PCTL90': spl_pctl90,
    'PCTL95': spl_pctl95,
    'PCTL98': spl_pctl98,
    'PCTL99': spl_pctl99
})

# Spectrosum representation:  
x = np.arange(len(OctaveBands['fc'])+1) 
barwidth = 1/9 #to adjust barwidth
xticks_str=[str(int(f//1)) for f in OctaveBands['fc']]
plt.ioff() # Turn interactive plotting off
fig = plt.figure(figsize=(8,8))
plt.bar(x[1:]-barwidth*5, spl_mean, color='gold', width=barwidth, align='edge', label='Mean')
plt.bar(x[1:]-barwidth*4, spl_pctl25, color='midnightblue', width=barwidth, align='edge', label='PCTL25th (%.1fs)' %(Tdur-Tdur*0.25))
plt.bar(x[1:]-barwidth*3, spl_pctl50, color='darkgreen', width=barwidth, align='edge', label='PCTL50th (%.1fs)' %(Tdur-Tdur*0.5))
plt.bar(x[1:]-barwidth*2, spl_pctl75, color='red', width=barwidth, align='edge', label='PCTL75th (%.1fs)' %(Tdur-Tdur*0.75))
plt.bar(x[1:]-barwidth, spl_pctl90, color='lime', width=barwidth, align='edge', label='PCTL90th (%.1fs)' %(Tdur-Tdur*0.9))
plt.bar(x[1:], spl_pctl95, color='aqua', width=barwidth, align='edge', label='PCTL95th (%.1fs)' %(Tdur-Tdur*0.95))
plt.bar(x[1:]+barwidth, spl_pctl98, color='fuchsia', width=barwidth, align='edge', label='PCTL98th (%.1fs)' %(Tdur-Tdur*0.98))
plt.bar(x[1:]+barwidth*2, spl_pctl99, color='lightpink', width=barwidth, align='edge', label='PCTL99th (%.1fs)' %(Tdur-Tdur*0.99))
plt.legend(ncol=2,loc='upper center', fontsize=FontSize-2)
plt.xticks(x[1:], xticks_str)
plt.xlabel('Frequency [Hz]', fontsize=FontSize)
plt.xticks(rotation=90)
plt.ylabel(r"SPL [dB re 1$\mu$Pa]", fontsize=FontSize)
plt.ylim(PSDmin,PSDmax)
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()       

# SPLdata_cum png saving: 
save_low_SPLdataCum_png = True
if save_low_SPLdataCum_png:
    plt.savefig(os.path.join(datapath2save,SPLcumName+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# SPLdata_cum csv saving: 
save_low_SPLdataCum_csv = True
if save_low_SPLdataCum_csv:
    spl_cum.to_csv(os.path.join(datapath2save,SPLcumName+'.csv'),index=False,header=True,sep =';') 
    

# %% Spectrosum PSD combined:
PSDcomparisson = FileName + '_SPLdata_mean'
plt.ioff() # Turn interactive plotting off
fig = plt.figure(figsize=(14,8))
plt.plot(fspect, psd_mean, color=[0.3, 0.75, 0.93], label='Low freq.', linewidth = 4)
plt.plot(fspect_high, psd_mean_high, color=[0.77, 0.43, 0.84], label='High freq.', linewidth = 4)
plt.axvline(f_antialiasign,color='black',linestyle='--',linewidth=3)
plt.legend(ncol=4, loc='upper center', fontsize=FontSize-2)
plt.ylabel(r"PSD [dB re 1$\mu$Pa$^2$/Hz]", fontsize=FontSize)
plt.ylim(PSDmin,PSDmax)
plt.grid(visible='on')
plt.xscale('log')
plt.xticks([10,100,1e3, 10e3, 100e3], ['10 Hz','100 Hz','1 kHz','10 kHz','100 kHz'])
plt.xlabel("Frequency", fontsize=FontSize)
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
# plt.show()     

# SPLdata_cum png saving: 
PSDcomparisson_png = True
if PSDcomparisson_png:
    plt.savefig(os.path.join(datapath2save,PSDcomparisson+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all') 
    
# %% Round2mult function:
def Round2mult(num, mult, method='up'):
    '''
    Round a number to the nearest multiple of a specified value, using ceiling (up) or flooring (down).

    Summary:
    This function takes a number and rounds it to the nearest multiple of a specified value, 
    either rounding up (ceiling) or down (flooring) based on the chosen method.

    Parameters:
    - num (float): The number to be rounded.
    - mult (int or float): The multiple to which 'num' should be rounded.
    - method (str): The rounding method to use ('up' for rounding up, 'down' for rounding down).
                    Default is 'ceil'.

    Returns:
    - float: The nearest multiple of 'mult' based on the specified rounding method.

    Example:
    num_rounded_up = Round2mult(num, mult, method='up')
    num_rounded_down = Round2mult(num, mult, method='down')
    
    Created/Last modified: 2024-11-01
    '''
    if method == 'up':
        return np.ceil(num / mult) * mult
    elif method == 'down':
        return np.floor(num / mult) * mult
    else:
        raise ValueError("Invalid method. Use 'up' or 'down'.")


# %% Full log Spectrogram:
freq_over = f_antialiasign
indx_low = np.where((PSD_low[1:,0] <= freq_over))[0][-1]+1
indx_high = np.where((PSD_high[1:,0] >= freq_over))[0][0]
# psd_matrix = np.concatenate((PSD_low[:indx_low,:], PSD_high[indx_high:,:]), axis=0)

fmax_ytick = np.max(PSD_high[indx_high:,:][1:,0])
fmax_ytick = Round2mult(fmax_ytick*1e-3, 5, method='up')
yticks_values,yticks_str = [10,100,1000,10000,fmax_ytick*1e3], ['10 Hz', '100 Hz', '1 kHz', '10 kHz', '%i kHz' %(fmax_ytick)]

# tspect = psd_matrix[0,1:]
# fspect = psd_matrix[1:,0]
# psd = psd_matrix[1:,1:]
PSDfull =  FileName + '_full_PSDdata'
plt.ioff() # Turn interactive plotting off
plt.figure(figsize=(14,8))
# plt.xticks(rotation=Xlabel_rotation)
ax=plt.gca()
xfmt=md.DateFormatter('%H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
c_map = plt.colormaps['jet']
# taxis2plt=[dt.datetime.utcfromtimestamp(ts) for ts in tspect]
# plt.pcolormesh(taxis2plt, fspect, psd, cmap=c_map, vmin=PSDmin, vmax=PSDmax)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in PSD_high[0,1:]]
plt.pcolormesh(taxis2plt, PSD_high[indx_high:,:][1:,0], PSD_high[indx_high:,:][1:,1:], cmap=c_map, vmin=PSDmin, vmax=PSDmax)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in PSD_low[0,1:]]
plt.pcolormesh(taxis2plt, PSD_low[:indx_low,:][1:,0], PSD_low[:indx_low,:][1:,1:], cmap=c_map, vmin=PSDmin, vmax=PSDmax)
plt.axhline(y=freq_over, color='black', linestyle='--', label='%i Hz' %freq_over)
plt.ylabel('Frequency')
# plt.ylim(fspect[0],fspect[-1])
plt.yscale('log')
plt.yticks(yticks_values,yticks_str)
# plt.xlim(pd.Timestamp(taxis2plt[0]).round('H'),pd.Timestamp(taxis2plt[-1]).round('H'))
plt.title(PSDfull)
cbar = plt.colorbar()
cbar.set_label(r'PSD [dB re 1$\mu$Pa$^2$/Hz]', rotation=270, verticalalignment='baseline')
plt.tight_layout()
# plt.show()

save_full_PSDdata_png = True
if save_full_PSDdata_png:
    plt.savefig(os.path.join(datapath2save,PSDfull+'.png'), bbox_inches='tight', dpi = 150)    
plt.close('all')

# %% dBsum function:
def dBsum(dBs, log_base=10, axis=0):
    '''
    Calculate the sum of an array of dB values in the logarithmic domain, supporting multi-dimensional arrays.

    Summary:
    This function converts an array of dB values to the linear scale, computes their sum along the specified axis,
    and converts the result back to dB. The `log_base` parameter allows flexibility for different types of dB calculations 
    (e.g., power ratio with log_base=10 or sound pressure with log_base=20).

    Parameters:
    - dBs (numpy.ndarray): An array of decibel values to be summed.
    - log_base (int or float): The logarithm base to use for the conversion. Default is 10.
    - axis (int): The axis along which to compute the sum. Default is 0 (sum across columns).

    Returns:
    - numpy.ndarray: The sum value in decibels along the specified axis.

    Example:
    total_dB = dBsum(np.array([60, 62, 58, 61]), log_base=20) # For 1D array
    total_dB = dBsum(np.array([[60, 62], [58, 61]]), log_base=20, axis=0) # For 2D array

    Created/Last modified: 2024-12-14
    '''
    # Convert the dB values to linear scale
    linear_values = 10 ** (dBs / log_base)
    
    # Compute the sum along the specified axis
    linear_sum = np.sum(linear_values, axis=axis)
    
    # Convert the result back to dB
    return log_base * np.log10(linear_sum)

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
        warnings.warn("WARNING from OctaveBandCalculation(): freq < fini. Solution: freq=fini")
    
    Nbands = 55
    fcs = (fini*((2**(octave))**(np.arange(0, Nbands))))
    for fc in fcs: 
        f_low = fc/np.sqrt(2**octave)
        f_up = fc*np.sqrt(2**octave)    
        if freq >= f_low and freq <= f_up :
            break
       
    return fc,f_low,f_up 

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
        CSVdata = pd.read_csv(CSVfile,index_col=None,header=None,sep=';')
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

# %% dBsum function:
def dBsum(dBs, log_base=10, axis=0):
    '''
    Calculate the sum of an array of dB values in the logarithmic domain, supporting multi-dimensional arrays.

    Summary:
    This function converts an array of dB values to the linear scale, computes their sum along the specified axis,
    and converts the result back to dB. The `log_base` parameter allows flexibility for different types of dB calculations 
    (e.g., power ratio with log_base=10 or sound pressure with log_base=20).

    Parameters:
    - dBs (numpy.ndarray): An array of decibel values to be summed.
    - log_base (int or float): The logarithm base to use for the conversion. Default is 10.
    - axis (int): The axis along which to compute the sum. Default is 0 (sum across columns).

    Returns:
    - numpy.ndarray: The sum value in decibels along the specified axis.

    Example:
    total_dB = dBsum(np.array([60, 62, 58, 61]), log_base=20) # For 1D array
    total_dB = dBsum(np.array([[60, 62], [58, 61]]), log_base=20, axis=0) # For 2D array

    Created/Last modified: 2024-12-14
    '''
    # Convert the dB values to linear scale
    linear_values = 10 ** (dBs / log_base)
    
    # Compute the sum along the specified axis
    linear_sum = np.sum(linear_values, axis=axis)
    
    # Convert the result back to dB
    return log_base * np.log10(linear_sum)

# %% 
CSVfile = r'C:\Users\Didac\Documents\ddietor\LT-Acoustic-Feature-Extractor\scripts\results_test\onde2_0_0_3_17_02_26_00h_15_low_SPLdata_cum.csv'
freq2study = np.array([0,30])
octave_band = 1/3

octaves = []
for freq in np.arange(freq2study[0], freq2study[1] + 1, 1):
    # print(freq)
    fc,_,_ = OctaveBandCalculation(freq,octave = octave_band)
    # print(fc)
    octaves.append(fc)
octaves = np.unique(np.array(octaves))  
spl_matrix = []
for freq in octaves:
    try:
        time_read, freq_read, spl_read = signalReaderSPLdata(CSVfile, freq, octave_band, plotter=0)
        spl_matrix.append(spl_read)
    except:
        print('ERROR for %.1f Hz freq.' %freq)
        pass
spl_matrix = np.array(spl_matrix)

spl_sum = dBsum(spl_matrix,log_base=20, axis=0)

# %% Analysis SPL for a single 1/3 octaves:
from scipy.signal import butter, filtfilt
XlabelFormat,XlabelRot = '%H:%M:%S',0
FontSize = 18
Tw = 60
Tw_mean = 120 #Tw to analyze
overlap = 0.5
thres = 10 #Criteria 1: peaks above SPL median in the window [dB]
Dyn_thres = 1-Tw/Tw_mean # Criteria 2: peaks above SPL percentile in the window [dB]

spl_sum_env = np.abs(scp.signal.hilbert(spl_sum))
def low_pass_filter(signal, cutoff_freq, sample_rate):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)
spl_sum_env_filt = low_pass_filter(spl_sum_env, cutoff_freq=100, sample_rate=650)  # Example cutoff frequency

spl_time = spl_sum_env_filt
t = tspect.copy()
fs_spect = 1/(t[1]-t[0])
Nw = np.ceil(fs_spect*Tw)
Nw_mean = int(np.ceil(fs_spect*Tw_mean))
if not Nw_mean % 2:
    #Nw_mean is not odd (impair)
    Nw_mean -= 1
    
# Cutting:
j = 0
Nsplit = Nw_mean
Nover = int(np.round(Nw_mean*overlap))
Npeaks=[]
SNRpeaks=[]
while j < len(spl_time):
    if j == 0:
        Nini = j
        Nend = Nini + Nsplit
    else:
        Nini = Nend - Nover
        Nend = Nini + Nsplit
    j = Nend
    if len(spl_time) < Nend:
        Nend = len(spl_time)
        j = len(spl_time)
    print('Splitting SPL: %.1f%%' %(100*Nend/len(spl_time)))   
    
    # Splitting the SPL
    spl_time_split = spl_time[Nini:Nend]
    t_split = t[Nini:Nend]
    
    # # Median value in dB: 
    Amp_median = np.median(spl_time_split)        
            
    # Percentile in dB: 
    threshold = np.percentile(spl_time_split,100*Dyn_thres)        
    
    # find peaks: 
    peaks_1, _ = scp.signal.find_peaks(spl_time_split, height=Amp_median+thres, threshold=None, distance=Nw, prominence=None, width=None)
    peaks_2, _ = scp.signal.find_peaks(spl_time_split, height=threshold, threshold=None, distance=Nw, prominence=None, width=None)
    
    plt.figure()
    plt.plot(t_split-t_split[0],spl_time_split)
    # plt.axhline(y=Amp_median,linestyle='--',color='red',linewidth=.7,label='Median: %.1f dB' %Amp_median)
    # plt.axhline(y=Amp_median+thres,linestyle='-',color='red',linewidth=.7,label='Threshold +%.1f dB' %thres)
    plt.axhline(y=threshold,linestyle='-',color='red',linewidth=.7,label='Threshold (%.1f%%): %.1f dB' %(Dyn_thres*100,threshold))
    # plt.plot((t_split-t_split[0])[peaks_1],spl_time_split[peaks_1],marker='v',linestyle='None',color='black',linewidth=.7,label='Peaks')
    # plt.title('%s\nSPL$_{1/3 octave}$ %.1f Hz\nSPLIT (%f:%f)' %(spect_name,f,t_split[0],t_split[-1]))
    plt.ylabel('SPL [dB re 1$\mu$Pa]')
    plt.xlabel('Time [s]')
    # plt.ylim(PSDmin,PSDmax)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    Npeaks.append(Nini + np.intersect1d(peaks_1,peaks_2))
    SNRpeaks.append(spl_time_split[np.intersect1d(peaks_1,peaks_2)]-Amp_median)
# Npeaks = np.unique(np.concatenate(Npeaks, axis=0 ))
Npeaks_combined = np.concatenate(Npeaks, axis=0)
Npeaks, indxs = np.unique(Npeaks_combined, return_index=True)
SNRpeaks_combined = np.concatenate(SNRpeaks, axis=0)
SNRpeaks = SNRpeaks_combined[indxs]

plt.figure(figsize=(14,6))
ax=plt.gca()
xfmt=md.DateFormatter(XlabelFormat)
ax.xaxis.set_major_formatter(xfmt)
taxis2plt=[dt.datetime.fromtimestamp(ts) for ts in tspect]
plt.plot(taxis2plt,spl_time,color='black',linewidth=2)
plt.plot(taxis2plt,spl_sum,color='grey',alpha=.4,linewidth=1)
plt.plot([taxis2plt[i] for i in Npeaks],spl_time[Npeaks],marker='v',linestyle='None',color='black',linewidth=.7,label='Detections: %i' %len(Npeaks))
# Annotate the SNRpeaks values at the corresponding Npeaks
for i, peak_index in enumerate(Npeaks):
    # Position of the annotation: time (x) and SPL value (y)
    time = taxis2plt[peak_index]  # Corresponding time for the peak
    snr_value = SNRpeaks[i]  # Corresponding SNR value
    # Annotate the SNR value at the peak
    ax.annotate(f'SNR: {snr_value:.1f} dB',  # Text with SNR value
                xy=(time, spl_time[peak_index]),  # Position of the peak
                xytext=(0, 5),  # Offset the annotation slightly (x, y)
                textcoords='offset points',  # Offset in points (relative to xy)
                fontsize=FontSize-4,  # Font size
                color='black',  # Color of the text
                weight='bold',  # Make the text bold
                ha='center',  # Horizontal alignment
                va='bottom')  # Vertical alignment (position the text above the point)

plt.title(r'%s. SPL$_{sum.}$\nSearching events between %.1f and %.1f Hz of %.1f s in %.1f s windows (overlap: %.1f). Th: %.1f dB + %.1f %%' %(FileName,freq2study[0],freq2study[1], Tw, Tw_mean,overlap,thres, Dyn_thres*100), fontsize=FontSize)
plt.ylabel(r'SPL [dB re 1$\mu$Pa]', fontsize=FontSize)
# plt.ylim(PSDmin,PSDmax)
plt.legend(fontsize=FontSize-2)
plt.xticks(rotation=0)
plt.tick_params(axis='both', which='major', labelsize=FontSize)
plt.tick_params(axis='both', which='minor', labelsize=FontSize)
plt.tight_layout()
plt.show()
# %%
