from pathlib import Path
import librosa
import numpy as np
import pandas as pd


# File Paths

AUDIO_PATH = Path("data/sample.wav")
SENTENCE_PATH = Path("data/sentences.csv")

OUTPUT_DIR = Path("outputs")
METADATA_OUTPUT_PATH = OUTPUT_DIR / "metadata_summary.csv"
SEGMENT_OUTPUT_PATH = OUTPUT_DIR / "detected_segments.csv"


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


# Function:     detect_non_silent_segments
# Inputs:       y | audio waveform (NumPy array)
#               sr | sampling rate of the audio waveform, y (int)
#               top_db | silence threshold in decibels (int)
# Outputs:      segments | detected non-silent audio segments (pandas DataFrame)
# Description:  Detects non-silent parts of the audio file using librosa.

def detect_non_silent_segments(y, sr, top_db):
    intervals = librosa.effects.split(y, top_db=top_db) # Split into non-silent intervals

    segment_rows = []

    for i, interval in enumerate(intervals):
        start_sample = interval[0]
        end_sample = interval[1]

        start_time = start_sample / sr
        end_time = end_sample / sr
        d = end_time - start_time

        segment_rows.append(
            {
                "segment_id": i + 1,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(d, 3),
            }
        )

    segments = pd.DataFrame(segment_rows)

    return segments


# Function:     label_segment_types
# Inputs:       segments | detected non-silent audio segments (pandas DataFrame)
# Outputs:      segments | detected segments with speech/surrogacy labels (pandas DataFrame)
# Description:  Labels alternating detected segments as speech and surrogacy.

def label_segment_types(segments):
    segment_types = []

    for i in range(len(segments)):
        if i % 2 == 0:
            segment_types.append("speech")
        else:
            segment_types.append("surrogacy")

    segments["segment_type"] = segment_types

    return segments


# Function:     save_detected_segments
# Inputs:       segments | detected non-silent audio segments (pandas DataFrame)
# Outputs:      None
# Description:  Saves detected segment boundaries to a CSV file.

def save_detected_segments(segments):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segments.to_csv(SEGMENT_OUTPUT_PATH, index=False)


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

    print("\n\r  Detecting non-silent speech/surrogacy segments...", end="")

    segments = detect_non_silent_segments(y, sr, 30)

    segments = label_segment_types(segments)

    save_detected_segments(segments)

    print("\r  Segments detected successfully:                  ")
    print(f"    - Number of detected segments: {len(segments)}")
    print(f"    - Output file: {SEGMENT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()