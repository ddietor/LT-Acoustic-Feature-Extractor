# %% Libs:
import json
import mutagen
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import warnings
from typing import Optional, Tuple, Dict

# %% wav_creator() function: 
def wav_creator(
    signal_uPa: np.ndarray,
    fs: int,
    filename: str,
    save_path: str = "./",
    Nbits: int = 32,
    FS_uPa: Optional[float] = None,
    metadata_str: Optional[str] = None,
) -> float:
    """
    Save a signal (in µPa) as a WAV file.

    Parameters
    ----------
    signal_uPa : np.ndarray
        Input signal in microPascal.
    fs : int
        Sampling frequency (Hz).
    filename : str
        Output filename (with or without .wav extension).
    save_path : str, optional
        Directory where the WAV file will be saved (default: current directory).
    Nbits : int, optional. Nbits must be 16, 24, or 32.
        Bit depth (default: 32).
    FS_uPa : float, optional
        Full-scale value in µPa. If None, it will be computed.
    metadata_str : str, optional
        Metadata string (e.g., JSON) to embed in the WAV file.

    Returns
    -------
    FS_uPa : float, optional
        Full-scale value in µPa to read again the signal in uPa.
    """

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if Nbits not in (16, 24, 32):
        raise ValueError("Nbits must be 16, 24, or 32")

    if not filename.lower().endswith(".wav"):
        filename += ".wav"

    # Compute FS_uPa if not provided
    if FS_uPa is None:
        FS_uPa = np.max(np.abs(signal_uPa)) * 1.1  # Add 10% headroom

    # Normalize and clip
    signal_norm = np.clip(signal_uPa / FS_uPa, -1, 1).astype(np.float32)

    # Quantize
    max_int = 2 ** (Nbits - 1) - 1
    if Nbits == 16:
        signal_bits = (signal_norm * max_int).astype(np.int16)
    elif Nbits == 24:
        # stored as int32 but limited to 24-bit range
        signal_bits = np.clip((signal_norm * max_int), -max_int, max_int).astype(np.int32)
    elif Nbits == 32:
        signal_bits = (signal_norm * max_int).astype(np.int32)

    # Save
    wav_path = save_path / filename
    wavfile.write(str(wav_path), fs, signal_bits)

    # Metadata
    if metadata_str is not None:
        audio_file = mutagen.File(str(wav_path))

        if audio_file is not None:
            if audio_file.tags is None:
                audio_file.add_tags()

            metadata_comment = mutagen.id3.TXXX(
                encoding=3,              # UTF-8
                desc="WAVmetadata",
                text=metadata_str
            )

            audio_file.tags.add(metadata_comment)
            audio_file.save()

    return FS_uPa

# %% wav_reader() function: 
def wav_reader(
    wavFile: str,
    Nbits: int = 32,
    FS_uPa: Optional[float] = None,
) -> Tuple[np.ndarray, int, Optional[str]]:
    """
    Read a WAV file and convert it back to µPa, optionally extracting metadata.

    Parameters
    ----------
    wavFile : str
        Path to WAV file.
    Nbits : int, optional
        Bit depth used during encoding (default: 32).
    FS_uPa : float, optional
        Full-scale value in µPa. If None, must be inferred externally.

    Returns
    -------
    Tuple[np.ndarray, int, Optional[str]]
        signal_uPa : np.ndarray
            Signal in microPascal.
        fs : int
            Sampling frequency.
        metadata_str : Optional[str]
            Extracted metadata string (if present).
    """

    wav_path = Path(wavFile)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, signal_samples = wavfile.read(str(wav_path))

    # Read WAV
    fs, signal_samples = wavfile.read(str(wav_path))

    # Normalize
    signal_norm = signal_samples.astype(np.float64) / (2 ** (Nbits - 1))

    # Convert to µPa
    if FS_uPa is None:
        raise ValueError("FS_uPa must be provided to reconstruct signal in µPa.")

    signal_uPa = signal_norm * FS_uPa

    # Read metadata
    metadata_str = None
    audio_file = mutagen.File(str(wav_path))

    if audio_file is not None and audio_file.tags is not None:
        for tag in audio_file.tags.values():
            # Look for your custom tag
            if hasattr(tag, "desc") and tag.desc == "WAVmetadata":
                metadata_str = tag.text[0] if isinstance(tag.text, list) else tag.text
                break

    return signal_uPa, fs, metadata_str

# %% wav_metadata_reader() function: 
def wav_metadata_reader(wavFile: str) -> Optional[Dict]:
    """
    Read metadata from a WAV file (TXXX tag).

    Parameters
    ----------
    wavFile : str
        Path to WAV file.

    Returns
    -------
    Optional[Dict]
        Metadata dictionary if found and valid JSON, otherwise None.
    """

    wav_path = Path(wavFile)

    audio_file = mutagen.File(str(wav_path))

    if audio_file is None or audio_file.tags is None:
        return None

    for tag in audio_file.tags.values():
        if hasattr(tag, "desc") and tag.desc == "WAVmetadata":
            metadata_str = tag.text[0] if isinstance(tag.text, list) else tag.text

            # Try parsing JSON
            try:
                return json.loads(metadata_str)
            except Exception:
                # If not JSON, return raw string
                return {"raw_metadata": metadata_str}

    return None