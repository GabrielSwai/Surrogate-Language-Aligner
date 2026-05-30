from pathlib import Path
import librosa
import numpy as np
import pympi


# File Paths

AUDIO_PATH = Path("data/sample.wav")
ELAN_PATH = Path("data/sample.eaf")

OUTPUT_DIR = Path("outputs")

TIER_NAME = "Surrogate_Transcription-txt-gbe"


# Function:     load_audio
# Inputs:       audio_path | file path of audio file to be loaded (Path)
# Outputs:      y | audio waveform (NumPy array)
#               sr | sampling rate of the audio waveform, y (int)
#               d | duration of the audio (float)
# Description:  Loads an audio file using librosa.

def load_audio(audio_path):
    # Load the audio file with its original sampling rate
    y, sr = librosa.load(audio_path, sr=None)

    # Get the duration of the audio file in seconds
    d = librosa.get_duration(y=y, sr=sr)

    # Return the waveform, sampling rate, and duration
    return y, sr, d


# Function:     load_elan
# Inputs:       elan_path | file path of ELAN file to be loaded (Path)
# Outputs:      eaf | loaded ELAN file object (Eaf)
# Description:  Loads an ELAN .eaf file so that its annotation tiers can be read/modified.

def load_elan(elan_path):
    # Convert Path to string so pympi can read it
    elan_path = str(elan_path)

    # Load the ELAN file
    eaf = pympi.Elan.Eaf(elan_path)

    # Return the loaded ELAN object
    return eaf


# Function:     extract_audio_segment
# Inputs:       y | audio waveform (NumPy array)
#               sr | sampling rate of the audio waveform, y (int)
#               start_time | start time of the segment in seconds (float)
#               end_time | end time of the segment in seconds (float)
# Outputs:      y_segment | extracted audio segment (NumPy array)
# Description:  Extracts a smaller audio segment from the full audio file using start and end times.

def extract_audio_segment(y, sr, start_time, end_time):
    # Convert start time from seconds to a sample index
    start_sample = int(start_time * sr)

    # Convert end time from seconds to a sample index
    end_sample = int(end_time * sr)

    # Slice the waveform to keep only the selected segment
    y_segment = y[start_sample:end_sample]

    # Return the extracted audio segment
    return y_segment


# Function:     get_elan_segments
# Inputs:       eaf | loaded ELAN file object
#               tier_name | name of ELAN tier to extract annotations from (str)
# Outputs:      segments | list of annotation intervals and labels from the specified tier (list)
# Description:  Extracts start times, end times, and annotation values from a given ELAN tier.

def get_elan_segments(eaf, tier_name):
    # Get all annotations from the specified ELAN tier
    annotations = eaf.get_annotation_data_for_tier(tier_name)

    # Create an empty list to store the cleaned segment data
    segments = []

    # Loop through each annotation in the tier
    for start_time, end_time, label in annotations:

        # Store each annotation as a dictionary
        segment = {
            "start": start_time / 1000,
            "end": end_time / 1000,
            "label": label
        }

        # Add the segment to the list
        segments.append(segment)

    # Return the list of extracted segments
    return segments


# Function:     detect_note_onsets
# Inputs:       y_segment | extracted audio segment (NumPy array)
#               sr | sampling rate of the audio segment (int)
# Outputs:      onset_times | detected onset times of notes in seconds (list)
# Description:  Detects the approximate start times of individual surrogate notes in the audio segment.

def detect_note_onsets(y_segment, sr):
    # Detect onset frames in the audio segment
    onset_frames = librosa.onset.onset_detect(y=y_segment, sr=sr)

    # Convert onset frames into times in seconds
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Convert NumPy array to a regular Python list
    onset_times = onset_times.tolist()

    # Return the detected onset times
    return onset_times


# Function:     make_note_intervals
# Inputs:       onset_times | detected onset times of notes in seconds (list)
#               segment_start | start time of the full segment in seconds (float)
#               segment_end | end time of the full segment in seconds (float)
# Outputs:      intervals | start and end times for each detected note (list)
# Description:  Converts note onset times into note intervals with start and end times.

def make_note_intervals(onset_times, segment_start, segment_end):
    # Initialize an empty list to store note intervals
    intervals = []

    # Convert relative onset times to absolute times
    absolute_onsets = [segment_start + onset for onset in onset_times]

    # If no onsets were detected, return an empty list
    if len(absolute_onsets) == 0:
        return intervals

    # Loop through each detected onset
    for i in range(len(absolute_onsets)):

        # The note starts at the current onset
        start = absolute_onsets[i]

        # The note ends at the next onset, unless this is the last note
        if i < len(absolute_onsets) - 1:
            end = absolute_onsets[i + 1]

        # The final note ends at the end of the full segment
        else:
            end = segment_end

        # Store the note interval
        intervals.append({
            "start": start,
            "end": end
        })

    # Return the list of note intervals
    return intervals


# Function:     extract_pitch
# Inputs:       y_segment | extracted audio segment (NumPy array)
#               sr | sampling rate of the audio segment (int)
#               intervals | start and end times for each detected note (list)
# Outputs:      pitch_features | pitch-related feature values for each note interval (list)
# Description:  Extracts acoustic pitch features from each detected note interval using spectral centroid.

def extract_pitch(y_segment, sr, intervals):
    # Create an empty list to store pitch features
    pitch_features = []

    # Loop through each note interval
    for interval in intervals:

        # Convert absolute start time to a time relative to the segment
        relative_start = interval["start"] - intervals[0]["start"]

        # Convert absolute end time to a time relative to the segment
        relative_end = interval["end"] - intervals[0]["start"]

        # Convert start time from seconds to samples
        start_sample = int(relative_start * sr)

        # Convert end time from seconds to samples
        end_sample = int(relative_end * sr)

        # Extract just this note from the segment
        y_note = y_segment[start_sample:end_sample]

        # Skip empty intervals
        if len(y_note) == 0:
            pitch_features.append(0)
            continue

        # Compute the spectral centroid for the note
        centroid = librosa.feature.spectral_centroid(y=y_note, sr=sr)

        # Average the centroid across the note
        avg_centroid = float(centroid.mean())

        # Store the pitch feature
        pitch_features.append(avg_centroid)

    # Return one pitch feature for each note
    return pitch_features


# Function:     classify_pitches
# Inputs:       pitch_features | pitch-related feature values for each note interval (list)
# Outputs:      surrogate_tones | H/L tone labels for each detected note (list)
# Description:  Classifies each note as high or low based on its extracted pitch feature.

def classify_pitches(pitch_features):
    # Initialize an empty list to store H/L labels
    surrogate_tones = []

    # If there are no pitch features, return an empty list
    if len(pitch_features) == 0:
        return surrogate_tones

    # Use the median pitch feature as the H/L cutoff
    threshold = np.median(pitch_features)

    # Loop through each pitch feature
    for feature in pitch_features:

        # Label notes above the threshold as high
        if feature >= threshold:
            surrogate_tones.append("H")

        # Label notes below the threshold as low
        else:
            surrogate_tones.append("L")

    # Return the classified H/L labels
    return surrogate_tones


# Function:     parse_tones
# Inputs:       tone_string | word-level tone patterns separated by spaces (str)
# Outputs:      tones | parsed word-level tone patterns (list)
# Description:  Converts a tone string like "L-H L-L" into a list of tone patterns.

def parse_tones(tone_string):
    return # tones


# Function:     tokenize_phrase
# Inputs:       phrase | Kinande phrase to be tokenized (str)
# Outputs:      words | word tokens from the phrase (list)
# Description:  Splits a Kinande phrase into word tokens using spaces.

def tokenize_phrase(phrase):
    return # words


# Function:     validate_melodies
# Inputs:       words | word tokens from the Kinande phrase (list)
#               tones | word-level tone patterns for the Kinande phrase (list)
# Outputs:      is_valid | whether the words and tones match correctly (bool)
# Description:  Checks that every word has exactly one corresponding tone pattern.

def validate_melodies(words, tones):
    return # is_valid


# Function:     make_word_sequence
# Inputs:       words | word tokens from the Kinande phrase (list)
#               tones | word-level tone patterns for the Kinande phrase (list)
# Outputs:      word_sequence | structured word and tone pattern data (list)
# Description:  Combines Kinande words and their tone patterns into a structured sequence.

def make_word_sequence(words, tones):
    return # word_sequence


# Function:     make_surrogate_sequence
# Inputs:       intervals | start and end times for each detected note (list)
#               surrogate_tones | H/L tone labels for each detected note (list)
# Outputs:      surrogate_sequence | structured surrogate note data (list)
# Description:  Combines note intervals and H/L tone labels into a structured surrogate sequence.

def make_surrogate_sequence(intervals, surrogate_tones):
    return # surrogate_sequence


# Function:     group_tones_by_word
# Inputs:       word_sequence | structured word and tone pattern data (list)
#               surrogate_sequence | structured surrogate note data (list)
# Outputs:      grouped_sequence | surrogate notes grouped by corresponding Kinande word (list)
# Description:  Groups individual surrogate notes into word-sized chunks based on each word's tone pattern length.

def group_tones_by_word(word_sequence, surrogate_sequence):
    return # grouped_sequence


# Function:     score_alignment
# Inputs:       spoken_tone | tone from the Kinande word sequence (str)
#               surrogate_tone | tone from the surrogate note sequence (str)
# Outputs:      score | numerical alignment score for the tone pair (int or float)
# Description:  Scores a tone match, mismatch, or gap during sequence alignment.

def score_alignment(spoken_tone, surrogate_tone):
    return # score


# Function:     align_sequences
# Inputs:       spoken_sequence | structured Kinande word/tone sequence (list)
#               surrogate_sequence | structured surrogate note sequence (list)
# Outputs:      alignment | optimal alignment between Kinande words and surrogate notes (list)
# Description:  Aligns the Kinande tone sequence with the surrogate tone sequence using a scoring algorithm.

def align_sequences(spoken_sequence, surrogate_sequence):
    return # alignment


# Function:     calculate_confidence
# Inputs:       alignment | alignment between Kinande words and surrogate notes (list)
# Outputs:      confidence | confidence score for the alignment (float)
# Description:  Calculates an overall confidence score based on matches, mismatches, and gaps.

def calculate_confidence(alignment):
    return # confidence


# Function:     create_textgrid
# Inputs:       alignment | alignment between Kinande words and surrogate notes (list)
#               output_path | file path where TextGrid should be saved (Path)
# Outputs:      None
# Description:  Creates a Praat TextGrid containing the aligned words, tones, and surrogate note intervals.

def create_textgrid(alignment, output_path):
    return None


# Function:     add_alignment
# Inputs:       eaf | loaded ELAN file object
#               alignment | alignment between Kinande words and surrogate notes (list)
#               tier_name | name of ELAN tier where alignment should be added (str)
# Outputs:      eaf | modified ELAN file object
# Description:  Adds the generated word-to-surrogate alignment to a specified ELAN tier.

def add_alignment(eaf, alignment, tier_name):
    return eaf


# Function:     save_elan
# Inputs:       eaf | modified ELAN file object
#               output_path | file path where modified ELAN file should be saved (Path)
# Outputs:      None
# Description:  Saves the modified ELAN file to disk.

def save_elan(eaf, output_path):
    return None


# Function:     flag_mismatches
# Inputs:       alignment | alignment between Kinande words and surrogate notes (list)
# Outputs:      flagged_alignment | alignment with mismatches marked (list)
# Description:  Marks places where the Kinande tone pattern and surrogate tone pattern do not match cleanly.

def flag_mismatches(alignment):
    return # flagged_alignment


# Function:     run_pipeline
# Inputs:       config | configuration object containing file paths, tier names, and settings
# Outputs:      alignment | final alignment between Kinande words and surrogate notes (list)
#               confidence | confidence score for the final alignment (float)
# Description:  Runs the full alignment workflow from input loading to output file creation.

def run_pipeline(config):
    return # alignment, confidence


# Function:     main
# Inputs:       None
# Outputs:      None
# Description:  Main project pipeline.

def main():
    print("=== Basic Input Functions Test ===\n")

    # Make sure the outputs folder exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the audio file
    print("\rLoading audio...", end="")
    y, sr, d = load_audio(AUDIO_PATH)

    # Print basic audio information
    print("\rAudio loaded successfully:")
    print(f"  - Audio path: {AUDIO_PATH}")
    print(f"  - Sample rate: {sr} Hz")
    print(f"  - Duration: {d:.2f} seconds")
    print()

    # Load the ELAN file
    print("\rLoading ELAN file...", end="")
    eaf = load_elan(ELAN_PATH)

    # Print basic ELAN information
    print("\rELAN file loaded successfully:")
    print(f"  - ELAN path: {ELAN_PATH}")
    print()

    # Get segments from one ELAN tier
    print("\rGetting ELAN segments...", end="")
    segments = get_elan_segments(eaf, TIER_NAME)

    # Print how many segments were found
    print(f"\rFound {len(segments)} segments in tier: {TIER_NAME}")

    # Print the first few segments for checking
    print("\nFirst few segments:")
    for segment in segments[:5]:
        print(f"  - {segment}")
    print()

    # Choose a test segment from the ELAN file
    if len(segments) > 0:

        # Use the first segment as a test
        start_time = segments[0]["start"]
        end_time = segments[0]["end"]

        # Extract that part of the audio
        print("\rExtracting first audio segment...", end="")
        y_segment = extract_audio_segment(y, sr, start_time, end_time)

        # Get segment duration
        segment_duration = librosa.get_duration(y=y_segment, sr=sr)

        # Print segment info
        print("\rAudio segment extracted successfully:")
        print(f"  - Start time: {start_time:.2f} seconds")
        print(f"  - End time: {end_time:.2f} seconds")
        print(f"  - Segment duration: {segment_duration:.2f} seconds")
        print(f"  - Segment samples: {len(y_segment)}")

    else:

        # Tell the user if there were no annotations
        print("No segments found, so no audio segment was extracted.")

    print()

    # Detect note onsets in the extracted segment
    print("\rDetecting note onsets...", end="")
    onset_times = detect_note_onsets(y_segment, sr)

    # Print detected onset information
    print("\rNote onsets detected successfully:")
    print(f"  - Number of onsets: {len(onset_times)}")
    print(f"  - First few onsets: {onset_times[:10]}")
    print()

    # Convert note onsets into note intervals
    print("\rMaking note intervals...", end="")
    intervals = make_note_intervals(onset_times, start_time, end_time)

    # Print interval information
    print("\rNote intervals created successfully:")
    print(f"  - Number of intervals: {len(intervals)}")
    print(f"  - First few intervals:")
    for interval in intervals[:10]:
        print(f"    - {interval}")
    print()

    # Extract pitch features for each note interval
    print("\rExtracting pitch features...", end="")
    pitch_features = extract_pitch(y_segment, sr, intervals)

    # Print pitch feature information
    print("\rPitch features extracted successfully:")
    print(f"  - Number of pitch features: {len(pitch_features)}")
    print(f"  - First few pitch features: {pitch_features[:10]}")
    print()

    # Classify each pitch feature as H or L
    print("\rClassifying pitches...", end="")
    surrogate_tones = classify_pitches(pitch_features)

    # Print classified H/L tone information
    print("\rPitches classified successfully:")
    print(f"  - Number of surrogate tones: {len(surrogate_tones)}")
    print(f"  - First few surrogate tones: {surrogate_tones[:20]}")

    print("\nTesting complete.")


if __name__ == "__main__":
    main()