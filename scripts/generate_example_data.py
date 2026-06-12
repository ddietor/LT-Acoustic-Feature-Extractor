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

    # # Generate example data
    # config_signal = {
    #     "duration": 300,                # [s]
    #     "fs": 192000,                   # [Hz]
    #     "earthquake_start": 5,          # [s]
    #     "earthquake_duration": 35,      # [s]
    #     "noise_std": 2.77e6,            # [uPa]
    #     "earthquake_gain": 5.4,         # [a.u.] dB = 20 *np.log10(earthquake_gain)
    #     "tone_50k_amplitude": 21,
    #     "tone_100k_amplitude": 64,
    #     "seed": 12,                     # Random seed for reproducibility   
    # }

    config_signal = {
        "duration": 300,                # [s]
        "fs": 192000,                   # [Hz]
        "fs_env": 100,                  # [Hz]
        "noise_std": 2.77e6,            # [uPa]
        "earthquake_amplitude": 14.06,  # [uPa]
        "tone_50k_amplitude": 9.89e3,   # [uPa]
        "tone_68k_amplitude": 8.94e3,   # [uPa]
        "tone_75k_amplitude": 10.46e3,  # [uPa]
        "seed": 12,                     # Random seed for reproducibility   
    }
    rng = np.random.default_rng(config_signal["seed"])
    fs = config_signal["fs"]
    duration = config_signal["duration"]
    noise_std = config_signal["noise_std"]
    N = int(fs * duration)
    t = np.arange(N) / fs
    noise_synthetic = rng.normal(loc=0,scale=noise_std,size=duration*fs)
    tone50 = config_signal["tone_50k_amplitude"] * np.sin(2*np.pi*50e3*t + rng.uniform(0, 2*np.pi))
    tone68 = config_signal["tone_68k_amplitude"] * np.sin(2*np.pi*68e3*t + rng.uniform(0, 2*np.pi))
    tone75 = config_signal["tone_75k_amplitude"] * np.sin(2*np.pi*75e3*t + rng.uniform(0, 2*np.pi))
    tones_signal = tone50 + tone68 + tone75
    raw = rng.standard_normal(N)
    sos = scp.signal.butter(4,[1, 40],btype="bandpass",fs=fs,output="sos")
    seismic_base = scp.signal.sosfiltfilt(sos, raw)
    seismic_base /= np.std(seismic_base)
    earthquake = np.zeros(N)
    envelope_read = np.loadtxt("/Users/ddietor/Documents/ddietor/LT-Acoustic-Feature-Extractor/scripts/envelope_smooth.txt")
    fs_env = config_signal["fs_env"]
    t_env = np.arange(len(envelope_read)) / fs_env
    envelope = np.interp(t, t_env, envelope_read,left=0.0,right=0.0)
    earthquake_uPa = config_signal["earthquake_amplitude"]
    earthquake = envelope * seismic_base * earthquake_uPa
    signal_uPa = (earthquake + noise_synthetic + tones_signal)

    # Genera una senyal en amplitud creixent en freqüència, just jo esperava el contrari (pink noise as underwater noise)

    # noise = config_signal["noise_std"] * rng.standard_normal(N)
    # idx_start = int(config_signal["earthquake_start"] * fs)
    # idx_end = int((config_signal["earthquake_start"] + config_signal["earthquake_duration"]) * fs)
    # idxs_len = idx_end - idx_start
    # raw_eq = rng.standard_normal(idxs_len)
    # def lowpass(signal, cutoff, fs, order=4):
    #     sos = scp.signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    #     return scp.signal.sosfiltfilt(sos, signal)
    # raw_eq = lowpass(raw_eq, 31, fs)
    # raw_eq = raw_eq / np.std(raw_eq) # Normalize to unit std
    # x = np.linspace(0, 1, idxs_len)
    # envelope = np.exp(-3*x)   # decaimiento tipo coda
    # envelope[:int(0.2*len(x))] *= np.linspace(0,1,int(0.2*len(x)))  # attack
    # envelope = envelope / np.max(envelope)
    # earthquake = np.zeros(N)
    # earthquake_segment = (config_signal["earthquake_gain"]*envelope*raw_eq)
    # earthquake[idx_start:idx_end] = earthquake_segment
    # tone50 = config_signal["tone_50k_amplitude"]*np.sin(2*np.pi*50e3*t)
    # tone100 = config_signal["tone_100k_amplitude"]*np.sin(2*np.pi*100e3*t)
    # signal_uPa = noise + earthquake + tone50 + tone100

    factor_scale = 0.375*np.sqrt(2)
    Nbits = 24                      # ACDC
    RVR_hydro = -172                # Received Voltage Response [dB re 1uPa/V]
    Pref = 1e-6                     # Ref. pressure [Pa]
    Vref = 1                        # Ref. voltage [V]
    FS_uPa = factor_scale * (Pref / Vref) * (1 / 10**(RVR_hydro/20)) * 1e6 #Full Scale in uPa

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