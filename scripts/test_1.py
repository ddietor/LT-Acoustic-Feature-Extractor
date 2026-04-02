# %% Libs:
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as md
# import pandas as pd
import scipy as scp

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

# %% Spectrogram HF:

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

# %%
