from pathlib import Path
import librosa
import numpy as np


# File Paths

AUDIO_PATH = Path("data/sample.wav")

OUTPUT_DIR = Path("outputs")


# Function:     load_audio
# Inputs:       audio_path | file path of audio file to be loaded (Path)
# Outputs:      y | audio waveform (NumPy array)
#               sr | sampling rate of the audio waveform, y (int)
#               d | duration of the audio (float)
# Description:  Loads an audio file using librosa.

def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None) # Get y and sr from the file
    d = librosa.get_duration(y=y, sr=sr) # Get d from the file

    return y, sr, d


# Function:     main
# Inputs:       None
# Outputs:      None
# Description:  Main project pipeline.

def main():
    print("=== SURROGATE LANGUAGE ALIGNER ===\n")

    print("\r  Loading audio and sentences...", end="")
    
    y, sr, d = load_audio(AUDIO_PATH)


if __name__ == "__main__":
    main()