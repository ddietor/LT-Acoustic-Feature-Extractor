"""
Generate synthetic acoustic data for demonstrating and testing the
LT Acoustic Feature Extractor workflow.
The generated signal contains:
    - Background noise
    - A synthetic earthquake-like event
    - Narrowband electronic interference tones
The output is fully reproducible when using the default random seed.

# Created on Thu Jun 11 2026 14:25:04 UTC
@author: ddietor
"""

import argparse
import logging
import os
import json
import numpy as np
import scipy as scp

from src.wav_io import wav_creator

SCRIPT_NAME = "generate_example_data"
VERSION = "1.0"

def run():
    """
    Generate a synthetic example WAV file.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set up argument parser
    argparser = argparse.ArgumentParser(
        description=(
            "Generate synthetic acoustic data for the LT Acoustic Feature Extractor.\n"
            "Example usage:\n"
            "  python -m scripts.generate_example_data --output_results ./results"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "--output_results",
        type=str,
        default="./results",
        help="Folder to store analysis results (default: ./WRSapplication_results).",
    )
    argparser.add_argument(
        "--filename",
        type=str,
        default="example_data",
        help="...",
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )

    # Parse arguments
    args = argparser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    else:
        logger.setLevel(logging.INFO)

    # Remove the output_results folder if it already exists to avoid mixing old and new results
    # if os.path.exists(args.output_results):
    #     shutil.rmtree(args.output_results)
    #     logger.info(f"Existing output_results ({args.output_results}) folder removed to ensure clean results.")

    # Create the output_results folder if not exists
    os.makedirs(args.output_results, exist_ok=True)

    config_signal = {
            "duration": 300,          # [s]
            "fs": 192000,             # [Hz]
            "fs_env": 250,            # [Hz]
            "noise_amp": 2.77e6,      # [uPa]
            "earthquake_ini": 5,      # [s]
            "earthquake_amp": 14.06e6,  # [uPa]
            "tone_50k_amp": 9.89e3,   # [uPa]
            "tone_68k_amp": 8.94e3,   # [uPa]
            "tone_75k_amp": 10.46e3,  # [uPa]
            "seed": 12,                     # Random seed for reproducibility   
        }
    
    factor_scale = 0.375*np.sqrt(2)
    Nbits = 24                      # ACDC
    RVR_hydro = -172                # Received Voltage Response [dB re 1uPa/V]
    Pref = 1e-6                     # Ref. pressure [Pa]
    Vref = 1                        # Ref. voltage [V]
    FS_uPa = factor_scale * (Pref / Vref) * (1 / 10**(RVR_hydro/20)) * 1e6 #Full Scale in uPa

    rng = np.random.default_rng(config_signal["seed"])
    fs = config_signal["fs"]
    duration = config_signal["duration"]
    Nsamples = int(fs * duration)
    t = np.arange(Nsamples) / fs
    white_noise= rng.normal(0,1,Nsamples)

    Xfft_white = np.fft.rfft(white_noise)
    white_f = np.fft.rfftfreq(Nsamples, 1/fs)
    white_f[0] = white_f[1]
    Xfft_pink = Xfft_white / np.sqrt(white_f)
    pink_noise = np.fft.irfft(Xfft_pink, n=Nsamples)
    pink_noise = pink_noise / np.max(np.abs(pink_noise))

    tone50 = np.sin(2*np.pi*50e3*t + rng.uniform(0, 2*np.pi)) * config_signal["tone_50k_amp"] / config_signal["tone_50k_amp"]
    tone68 = np.sin(2*np.pi*68e3*t + rng.uniform(0, 2*np.pi)) * config_signal["tone_68k_amp"] / config_signal["tone_68k_amp"]
    tone75 = np.sin(2*np.pi*75e3*t + rng.uniform(0, 2*np.pi)) * config_signal["tone_75k_amp"] / config_signal["tone_75k_amp"]
    tones_signal = tone50 + tone68 + tone75
    tones_signal /= 3.0

    envelope_read = np.loadtxt("./scripts/earthqueake_envelope.txt")
    fs_env = config_signal["fs_env"]
    t_env = np.arange(len(envelope_read)) / fs_env
    t_env_fs = np.arange(0, t_env[-1], 1 / fs) 
    envelope = np.interp(t_env_fs, t_env, envelope_read,left=0.0,right=0.0)
    envelope = envelope[int(t_env_fs[0]+1 * fs):int(t_env_fs[-1]-1 * fs)]
    envelope = envelope / np.max(np.abs(envelope))
    # t_env_fs = t_env_fs[int(t_env_fs[0]+1 * fs):int(t_env_fs[-1]-1 * fs)]

    earthquake_noise = rng.standard_normal(len(envelope))
    sos = scp.signal.butter(4,[1, 40],btype="bandpass",fs=fs,output="sos")
    earthquake_noise = scp.signal.sosfiltfilt(sos, earthquake_noise)
    earthquake_noise = earthquake_noise / np.max(np.abs(earthquake_noise))
    earthquake = envelope * earthquake_noise 
    earthquake = earthquake / np.max(np.abs(earthquake))
    earthquake *= (config_signal["earthquake_amp"] / FS_uPa)

    ToA_earthquake = config_signal["earthquake_ini"]
    Nini = int(ToA_earthquake * fs)
    Nend = Nini + len(earthquake)

    signal_rel = pink_noise.copy()
    signal_rel[Nini:Nend] += earthquake
    signal_rel += tones_signal
    signal_rel /= np.max(np.abs(signal_rel))
    # signal_uPa = signal_rel * FS_uPa 
    signal_uPa = signal_rel

    metadata = {
        "description": "Signal Test",
        "duration_s": duration,
        "fs_Hz": fs,
        "Nbits": Nbits,
        "FS_uPa": FS_uPa,
    }

    wav_creator(
        signal_uPa,
        fs,
        filename="example_data.wav",
        save_path=args.output_results,
        FS_uPa=None,
        metadata_str = json.dumps(metadata)
    )

    logger.info(f"...{SCRIPT_NAME} finalize!")


if __name__ == "__main__":
    run()