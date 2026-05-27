from pathlib import Path
import librosa
import pandas as pd


# File Paths

AUDIO_PATH = Path("data/sample.wav")
SENTENCE_PATH = Path("data/sentences.csv")
OUTPUT_DIR = Path("outputs")
METADATA_OUTPUT_PATH = OUTPUT_DIR / "metadata_summary.csv"


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


# Function:     load_sentences
# Inputs:       sentence_path | file path of sentence file to be loaded (Path)
# Outputs:      sentences | sentences from the file (pandas DataFrame)
# Description:  Loads elicitation sentences using pandas.

def load_sentences(sentence_path):
    sentences = pd.read_csv(sentence_path) # Convert CSV to pandas DataFrame

    return sentences


# Function:     save_metadata_summary
# Inputs:       audio_path | file path of audio file to be loaded (Path)
#               sentences | sentences from the file (pandas DataFrame)
#               sr | sampling rate of the audio waveform, y (int)
#               d | duration of the audio (float)
# Outputs:      None
# Description:  Saves metadata summary from the audio file and sentence list.

def save_metadata_summary(audio_path, sentences, sr, d):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Make the output directory if it doesn't exist. `parents=True` stops error if a parent doesn't exist; `exist_ok=True` stops error if it already exists.

    summary = pd.DataFrame( # Make a pandas DataFrame with metadata summary
        [
            {
                "audio_file": str(audio_path),
                "sentence_count": len(sentences),
                "sample_rate": sr,
                "duration_seconds": round(d, 2),
            }
        ]
    )

    summary.to_csv(METADATA_OUTPUT_PATH, index=False) # Save summary to CSV. `index=False` stops DataFrame index from appearing in CSV.


# Function:     main
# Inputs:       None
# Outputs:      None
# Description:  Main project pipeline.

def main():

    print("=== SURROGATE LANGUAGE ALIGNER ===\n")

    print("\r  Loading audio and sentences...", end="")
    
    y, sr, d = load_audio(AUDIO_PATH)
    sentences = load_sentences(SENTENCE_PATH)

    print("\r  Audio loaded successfully:    ")
    print(f"    - Audio file:  {AUDIO_PATH}")
    print(f"    - Sample rate: {sr}Hz")
    print(f"    - Duration:    {d:.2f} seconds")

    print("\n  Sentences loaded successfully:")
    print(f"    - Number of experiment sentences: {len(sentences)}")

    save_metadata_summary(
        audio_path=AUDIO_PATH,
        sentences=sentences,
        sr=sr,
        d=d,
    )

    print("\n  Metadata summary saved successfully:")
    print(f"    - Output file: {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()