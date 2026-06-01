from pathlib import Path
import librosa
import numpy as np
import pympi
import parselmouth


# Constants

AUDIO_PATH = Path(f"data/sample2.wav")
ELAN_PATH = Path(f"data/sample2.eaf")

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
    onset_frames = librosa.onset.onset_detect(
        y=y_segment,
        sr=sr,
        units="frames",
        backtrack=True,
        pre_max=10,
        post_max=10,
        pre_avg=20,
        post_avg=20,
        delta=0.2,
        wait=8
    )

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
#               low_min | 
# Outputs:      pitch_features | pitch-related feature values for each note interval (list)
# Description:  Extracts acoustic pitch features from each detected note interval using magnitude spectrum.

def extract_pitch(y_segment, sr, intervals, segment_start):
    # Create an empty list to store one pitch value per note
    pitch_features = []

    # Convert NumPy audio into a Parselmouth Sound object
    sound = parselmouth.Sound(y_segment, sampling_frequency=sr)

    # Run Praat's pitch tracker
    pitch = sound.to_pitch_ac(
        time_step=None,
        pitch_floor=50,
        max_number_of_candidates=15,
        very_accurate=False,
        silence_threshold=0.02,
        voicing_threshold=0.1,
        octave_cost=0.15,
        octave_jump_cost=0.5,
        voiced_unvoiced_cost=0.14,
        pitch_ceiling=800
    )

    # Loop through each note interval
    for interval in intervals:

        # Convert absolute times to segment-relative times
        relative_start = interval["start"] - segment_start
        relative_end = interval["end"] - segment_start

        # Get note duration
        note_duration = relative_end - relative_start

        # Skip the noisy attack
        analysis_start = relative_start + 0.3 * note_duration

        # Avoid the very end of the decay
        analysis_end = relative_start + 0.60 * note_duration

        # Get pitch values inside this note interval
        times = pitch.xs()
        values = pitch.selected_array["frequency"]

        # Keep only pitch values inside this note
        mask = (times >= analysis_start) & (times <= analysis_end)

        # Remove unvoiced frames, which Praat marks as 0
        note_pitches = values[mask]
        note_pitches = note_pitches[note_pitches > 0]

        # If no pitch was detected, store 0
        if len(note_pitches) == 0:
            pitch_features.append(0)
            continue

        # Use the median pitch for stability
        note_pitch = float(np.median(note_pitches))

        # Store one pitch value for this note
        pitch_features.append(note_pitch)

    # Return one pitch value per note interval
    return pitch_features


# Function:     extract_shape
# Inputs:       y_segment | extracted audio segment (NumPy array)
#               sr | sampling rate of the audio segment (int)
#               intervals | start and end times for each detected note (list)
#               segment_start | start time of the full segment in seconds (float)
# Outputs:      shape_features | shortness/decay score for each detected note (list)
# Description:  Extracts note-shape features based on how quickly each strike decays.

def extract_shape(
    y_segment,
    sr,
    intervals,
    segment_start,
    shape_window_seconds=0.1,
    frame_ms=8,
    hop_ms=2,
    threshold_ratio=0.52,
    max_tail_time=0.25
):

    # Create an empty list to store shape scores
    shape_features = []

    # Convert frame size from milliseconds to samples
    frame_length = max(1, int((frame_ms / 1000) * sr))

    # Convert hop size from milliseconds to samples
    hop_length = max(1, int((hop_ms / 1000) * sr))

    # Loop through each note interval
    for interval in intervals:

        # Convert absolute start time to segment-relative time
        relative_start = interval["start"] - segment_start

        # Convert start time to samples
        start_sample = int(relative_start * sr)

        # Use a fixed window after onset
        end_sample = min(
            start_sample + int(shape_window_seconds * sr),
            len(y_segment)
        )

        # Extract note window
        y_note = y_segment[start_sample:end_sample]

        # Skip empty notes
        if len(y_note) == 0:
            shape_features.append(0)
            continue

        # Compute RMS envelope
        rms = librosa.feature.rms(
            y=y_note,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Skip empty RMS
        if len(rms) == 0:
            shape_features.append(0)
            continue

        # Get peak RMS
        peak = np.max(rms)

        # Avoid division by zero
        if peak == 0:
            shape_features.append(0)
            continue

        # Normalize RMS by peak
        rms = rms / peak

        # Search for peak only near the beginning
        peak_search_frames = max(1, int(0.05 / (hop_ms / 1000)))

        # Prevent search window from exceeding RMS length
        peak_search_frames = min(peak_search_frames, len(rms))

        # Find peak index in early part
        peak_index = int(np.argmax(rms[:peak_search_frames]))

        # Keep decay after peak
        decay = rms[peak_index:]

        # Skip empty decay
        if len(decay) == 0:
            shape_features.append(0)
            continue

        # Find frames still above threshold
        active = np.where(decay >= threshold_ratio)[0]

        # If no active frames, tail time is zero
        if len(active) == 0:
            tail_time = 0

        # Otherwise get last active frame time
        else:
            tail_time = (active[-1] * hop_length) / sr

        # Convert tail time to H-like shortness score
        tail_shortness = 1.0 - (tail_time / max_tail_time)

        # Clamp to 0-1
        tail_shortness = max(0, min(tail_shortness, 1))

        # Define early region after peak
        early_end = min(len(decay), max(2, int(0.05 / (hop_ms / 1000))))

        # Define late region after early region
        late_start = min(len(decay), max(2, int(0.10 / (hop_ms / 1000))))

        # Compute early energy
        early_energy = np.mean(decay[:early_end])

        # Compute late energy
        if late_start >= len(decay):
            late_energy = 0
        else:
            late_energy = np.mean(decay[late_start:])

        # Lower tail energy is more H-like
        tail_energy_ratio = late_energy / (early_energy + 1e-9)

        # Convert to shortness score
        energy_shortness = 1.0 - tail_energy_ratio

        # Clamp to 0-1
        energy_shortness = max(0, min(energy_shortness, 1))

        # Estimate decay slope on log RMS
        x = np.arange(len(decay))

        # Avoid log(0)
        y = np.log(decay + 1e-6)

        # Fit a simple line to log-decay
        if len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0

        # More negative slope means faster decay
        slope_shortness = min(1, max(0, -slope * 10))

        # Combine cues into one shape score
        shape_score = (
            0.45 * tail_shortness
            + 0.35 * energy_shortness
            + 0.20 * slope_shortness
        )

        # Store shape score
        shape_features.append(float(shape_score))

    # Return one shape score per note
    return shape_features


# Function:     classify_pitches
# Inputs:       pitch_features | pitch-related feature values for each note interval (list)
# Outputs:      surrogate_tones | H/L tone labels for each detected note (list)
# Description:  Classifies each note as high or low based on its extracted pitch feature and strike shape.

def classify_pitches(pitch_features, shape_features):

    # If there are no pitch features, return an empty list
    if len(pitch_features) == 0:
        return []

    # If no shape features were given, use zeros
    if shape_features is None:
        shape_features = [0] * len(pitch_features)

    # Make sure both lists match
    if len(pitch_features) != len(shape_features):
        raise ValueError("pitch_features and shape_features must have the same length.")

    # Convert pitches to NumPy array
    pitches = np.array(pitch_features, dtype=float)

    # Convert shape features to NumPy array
    shapes = np.array(shape_features, dtype=float)

    # Normalize shape features
    if np.max(shapes) == np.min(shapes):
        shape_norm = np.zeros_like(shapes)
    else:
        shape_norm = (shapes - np.min(shapes)) / (np.max(shapes) - np.min(shapes))

    # Keep only valid pitch values
    valid_pitches = pitches[pitches > 0]

    # If no valid pitches exist, classify by shape only
    if len(valid_pitches) == 0:
        boundary = np.median(shape_norm)
        return ["H" if s >= boundary else "L" for s in shape_norm]

    # Initialize centers
    pitch_boundary = 120.0

    # Create empty list for labels
    surrogate_tones = []

    # Classify each note
    for pitch, shape in zip(pitches, shape_norm):
        # If pitch exists, classify using pitch only
        if pitch > 0:
            # Compare pitch directly to boundary
            if pitch >= pitch_boundary:
                surrogate_tones.append("H")
            else:
                surrogate_tones.append("L")

        # If pitch failed, assume H, but let shape override if it is strongly L-like
        else:
            # Very low shortness means long/ringy tail, so mark as L
            if shape <= 0.4:
                surrogate_tones.append("L")

            # Otherwise default to H
            else:
                surrogate_tones.append("H")

    # Return H/L labels
    return surrogate_tones


# Function:     parse_tones
# Inputs:       tone_string | spoken phrase (str)
# Outputs:      tones | parsed word-level tone patterns (list)
# Description:  Converts a phrase like "Tukándisyá tasáta kwȇ?" into a list of tone patterns like ["L-H-L-H", "L-H-L", "HL"].

def parse_tones(tone_string):
    # Vowels with high tone
    h_vowels = ["á", "é", "í", "ó", "ú", "Á", "É", "Í", "Ó", "Ú"]

    # Vowels with low tone
    l_vowels = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]

    # Vowels with falling tone
    f_vowels = ["â", "ê", "î", "ô", "û", "ȃ", "ȇ", "ȋ", "ȏ", "ȗ",
                "Â", "Ê", "Î", "Ô", "Û", "Ȃ", "Ȇ", "Ȋ", "Ȏ", "Ȗ"]

    # Create an empty list for word-level tone patterns
    tones = []

    # Split the phrase into words
    words = tone_string.split()

    # Loop through each word
    for word in words:

        # Create an empty list for this word's tones
        word_tones = []

        # Loop through each character in the word
        for char in word:

            # Check for high-toned vowels
            if char in h_vowels:
                word_tones.append("H")

            # Check for low-toned vowels
            elif char in l_vowels:
                word_tones.append("L")

            # Check for falling-toned vowels
            elif char in f_vowels:
                word_tones.append("HL")

        # Join this word's tones with hyphens
        tone_pattern = "-".join(word_tones)

        # Add this word's tone pattern to the list
        tones.append(tone_pattern)

    # Return one tone pattern per word
    return tones


# Function:     tokenize_phrase
# Inputs:       phrase | Kinande phrase to be tokenized (str)
# Outputs:      words | word tokens from the phrase (list)
# Description:  Splits a Kinande phrase into word tokens using spaces.

def tokenize_phrase(phrase):
    # Split the phrase into words using spaces
    words = phrase.split()

    # Return the list of word tokens
    return words


# Function:     validate_melodies
# Inputs:       words | word tokens from the Kinande phrase (list)
#               tones | word-level tone patterns for the Kinande phrase (list)
# Outputs:      is_valid | whether the words and tones match correctly (bool)
# Description:  Checks that every word has exactly one corresponding tone pattern.

def validate_melodies(words, tones):
    # Check whether the number of words matches the number of tone patterns
    is_valid = len(words) == len(tones)

    # If the counts do not match, print a helpful message
    if not is_valid:
        print("Word/tone mismatch:")
        print(f"  - Number of words: {len(words)}")
        print(f"  - Number of tone patterns: {len(tones)}")

    # Return whether the melodies are valid
    return is_valid


# Function:     make_word_sequence
# Inputs:       words | word tokens from the Kinande phrase (list)
#               tones | word-level tone patterns for the Kinande phrase (list)
# Outputs:      word_sequence | structured word and tone pattern data (list)
# Description:  Combines Kinande words and their tone patterns into a structured sequence.

def make_word_sequence(words, tones):
    # Create an empty list to store word objects
    word_sequence = []

    # Loop through each word and its matching tone pattern
    for i, (word, tone_pattern) in enumerate(zip(words, tones)):

        # Split the word-level tone pattern into individual tones
        tone_list = tone_pattern.split("-")

        # Store the word and its tone information together
        word_data = {
            "index": i,
            "word": word,
            "tone_pattern": tone_pattern,
            "tones": tone_list
        }

        # Add this word object to the sequence
        word_sequence.append(word_data)

    # Return the structured word sequence
    return word_sequence


# Function:     make_surrogate_sequence
# Inputs:       intervals | start and end times for each detected note (list)
#               surrogate_tones | H/L tone labels for each detected note (list)
# Outputs:      surrogate_sequence | structured surrogate note data (list)
# Description:  Combines note intervals and H/L tone labels into a structured surrogate sequence.

def make_surrogate_sequence(intervals, surrogate_tones):
    # Create an empty list to store surrogate note objects
    surrogate_sequence = []

    # Loop through each interval and its matching surrogate tone
    for i, (interval, tone) in enumerate(zip(intervals, surrogate_tones)):

        # Store the note interval and tone together
        note_data = {
            "index": i,
            "start": interval["start"],
            "end": interval["end"],
            "tone": tone
        }

        # Add this note object to the surrogate sequence
        surrogate_sequence.append(note_data)

    # Return the structured surrogate sequence
    return surrogate_sequence


# Function:     group_tones_by_word
# Inputs:       word_sequence | structured word and tone pattern data (list)
#               surrogate_sequence | structured surrogate note data (list)
# Outputs:      grouped_sequence | surrogate notes grouped by corresponding Kinande word (list)
# Description:  Groups individual surrogate notes into word-sized chunks based on each word's tone pattern length.

def group_tones_by_word(word_sequence, surrogate_sequence):
    # Create an empty list to store grouped word-level alignments
    grouped_sequence = []

    # Keep track of where we are in the surrogate note sequence
    surrogate_index = 0

    # Loop through each Kinande word
    for word_data in word_sequence:

        # Get the tones for this word
        word_tones = word_data["tones"]

        # Count how many surrogate notes this word should use
        tone_count = len(word_tones)

        # Get the matching chunk of surrogate notes
        surrogate_chunk = surrogate_sequence[surrogate_index:surrogate_index + tone_count]

        # Move the surrogate index forward
        surrogate_index += tone_count

        # Get the surrogate tones from this chunk
        surrogate_tones = [note["tone"] for note in surrogate_chunk]

        # If there are notes in the chunk, use their start and end times
        if len(surrogate_chunk) > 0:
            start = surrogate_chunk[0]["start"]
            end = surrogate_chunk[-1]["end"]

        # If there are no notes, leave times empty
        else:
            start = None
            end = None

        # Store the word and its grouped surrogate notes
        grouped_data = {
            "index": word_data["index"],
            "word": word_data["word"],
            "spoken_tone_pattern": word_data["tone_pattern"],
            "spoken_tones": word_tones,
            "surrogate_tone_pattern": "-".join(surrogate_tones),
            "surrogate_tones": surrogate_tones,
            "surrogate_notes": surrogate_chunk,
            "start": start,
            "end": end
        }

        # Add this grouped word object to the output
        grouped_sequence.append(grouped_data)

    # Return the grouped word-level sequence
    return grouped_sequence


# Function:     score_alignment
# Inputs:       spoken_tone | tone from the Kinande word sequence (str)
#               surrogate_tone | tone from the surrogate note sequence (str)
# Outputs:      score | numerical alignment score for the tone pair (int or float)
# Description:  Scores a tone match, mismatch, or gap during sequence alignment.

def score_alignment(spoken_tone, surrogate_tone):
    # If either tone is missing, treat it as a gap
    if spoken_tone is None or surrogate_tone is None:
        score = -2

    # Give a positive score for an exact tone match
    elif spoken_tone == surrogate_tone:
        score = 2

    # Give a smaller penalty for a mismatch
    else:
        score = -1

    # Return the alignment score
    return score


# Function:     align_sequences
# Inputs:       spoken_sequence | structured Kinande word/tone sequence (list)
#               surrogate_sequence | structured surrogate note sequence (list)
# Outputs:      alignment | optimal alignment between Kinande words and surrogate notes (list)
# Description:  Aligns the Kinande tone sequence with the surrogate tone sequence using a scoring algorithm.

def align_sequences(spoken_sequence, surrogate_sequence):

    # Flatten the spoken word sequence into individual tone objects
    spoken_tones = []
    for word_data in spoken_sequence:

        # Loop through each tone in this word
        for tone_index, tone in enumerate(word_data["tones"]):

            # Store the tone with its word information
            spoken_tones.append({
                "word_index": word_data["index"],
                "word": word_data["word"],
                "tone_index": tone_index,
                "tone": tone
            })

    # Flatten the surrogate sequence into individual tone objects
    surrogate_tones = []
    for note_data in surrogate_sequence:

        # Store the surrogate tone with its timing information
        surrogate_tones.append({
            "note_index": note_data["index"],
            "start": note_data["start"],
            "end": note_data["end"],
            "tone": note_data["tone"]
        })

    # Get sequence lengths
    n = len(spoken_tones)
    m = len(surrogate_tones)

    # Create score matrix
    score_matrix = np.zeros((n + 1, m + 1))

    # Create backpointer matrix
    backpointer_matrix = [[None for _ in range(m + 1)] for _ in range(n + 1)]

    # Fill first column with gap penalties
    for i in range(1, n + 1):
        score_matrix[i][0] = score_matrix[i - 1][0] + score_alignment(spoken_tones[i - 1]["tone"], None)
        backpointer_matrix[i][0] = "up"

    # Fill first row with gap penalties
    for j in range(1, m + 1):
        score_matrix[0][j] = score_matrix[0][j - 1] + score_alignment(None, surrogate_tones[j - 1]["tone"])
        backpointer_matrix[0][j] = "left"

    # Fill the rest of the matrix
    for i in range(1, n + 1):

        # Loop through surrogate tones
        for j in range(1, m + 1):

            # Score a spoken tone aligned to a surrogate tone
            diagonal_score = score_matrix[i - 1][j - 1] + score_alignment(
                spoken_tones[i - 1]["tone"],
                surrogate_tones[j - 1]["tone"]
            )

            # Score a spoken tone aligned to a gap
            up_score = score_matrix[i - 1][j] + score_alignment(
                spoken_tones[i - 1]["tone"],
                None
            )

            # Score a surrogate tone aligned to a gap
            left_score = score_matrix[i][j - 1] + score_alignment(
                None,
                surrogate_tones[j - 1]["tone"]
            )

            # Choose the best score
            best_score = max(diagonal_score, up_score, left_score)

            # Store the best score
            score_matrix[i][j] = best_score

            # Store the direction that produced the best score
            if best_score == diagonal_score:
                backpointer_matrix[i][j] = "diagonal"
            elif best_score == up_score:
                backpointer_matrix[i][j] = "up"
            else:
                backpointer_matrix[i][j] = "left"

    # Create an empty alignment list
    alignment = []

    # Start traceback from bottom-right corner
    i = n
    j = m

    # Trace back until both sequences are exhausted
    while i > 0 or j > 0:

        # Get traceback direction
        direction = backpointer_matrix[i][j]

        # Match/mismatch case
        if direction == "diagonal":

            # Get current spoken tone
            spoken = spoken_tones[i - 1]

            # Get current surrogate tone
            surrogate = surrogate_tones[j - 1]

            # Add aligned pair
            alignment.append({
                "spoken": spoken,
                "surrogate": surrogate,
                "spoken_tone": spoken["tone"],
                "surrogate_tone": surrogate["tone"],
                "start": surrogate["start"],
                "end": surrogate["end"],
                "score": score_alignment(spoken["tone"], surrogate["tone"])
            })

            # Move diagonally
            i -= 1
            j -= 1

        # Spoken tone aligned to gap
        elif direction == "up":

            # Get current spoken tone
            spoken = spoken_tones[i - 1]

            # Add gap alignment
            alignment.append({
                "spoken": spoken,
                "surrogate": None,
                "spoken_tone": spoken["tone"],
                "surrogate_tone": None,
                "start": None,
                "end": None,
                "score": score_alignment(spoken["tone"], None)
            })

            # Move up
            i -= 1

        # Surrogate tone aligned to gap
        elif direction == "left":

            # Get current surrogate tone
            surrogate = surrogate_tones[j - 1]

            # Add gap alignment
            alignment.append({
                "spoken": None,
                "surrogate": surrogate,
                "spoken_tone": None,
                "surrogate_tone": surrogate["tone"],
                "start": surrogate["start"],
                "end": surrogate["end"],
                "score": score_alignment(None, surrogate["tone"])
            })

            # Move left
            j -= 1

        # Safety fallback
        else:
            break

    # Reverse alignment because traceback builds it backwards
    alignment.reverse()

    # Return the final alignment
    return alignment


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
    pitch_features = extract_pitch(y_segment, sr, intervals, start_time)

    # Print pitch feature information
    print("\rPitch features extracted successfully:")
    print(f"  - Number of pitch features: {len(pitch_features)}")
    print(f"  - First few pitch features: {pitch_features[:10]}")
    print()

    # Extract shape features for each note interval
    print("\rExtracting shape features...", end="")
    shape_features = extract_shape(y_segment, sr, intervals, start_time)

    # Classify each pitch feature as H or L
    print("\rClassifying pitches...", end="")
    surrogate_tones = classify_pitches(pitch_features, shape_features)

    # Print classified H/L tone information
    print("\rPitches classified successfully:")
    print(f"  - Number of surrogate tones: {len(surrogate_tones)}")
    print(f"  - First few surrogate tones: {surrogate_tones[:20]}")

    # Print features with labels
    for feature, tone in zip(pitch_features, surrogate_tones):
        print(f"{feature:.3f} -> {tone}")
    
    # Print accuracy
    if AUDIO_PATH == Path(f"data/sample.wav"):
        correct_tones = ['L', 'L', 'L', 'L', 'H', 'L', 'H', 'L', 'L', 'L', 'L']
    elif AUDIO_PATH == Path(f"data/sample2.wav"):
        correct_tones = ['L', 'L', 'L', 'H', 'L', 'H', 'L', 'L', 'L', 'L', 'H', 'H', 'H', 'H', 'H']
    print(f"\nCorrect surrogate tones: {correct_tones}")
    print(f"\nOutput:                  {surrogate_tones}")
    correct = 0
    for i in range(len(correct_tones)):
        if correct_tones[i] == surrogate_tones[i]:
            correct += 1
    print(f"\nAccuracy: {correct/len(correct_tones):.2f}")
    print()

    # Test phrase tokenization and tone parsing
    print("\rTesting phrase and tone parsing...", end="")

    # Use a small test phrase with tone markings
    if AUDIO_PATH == Path(f"data/sample.wav"):
        test_phrase = "Lebay’ ebitú, yikátak’ eriwȃ"
    if AUDIO_PATH == Path(f"data/sample2.wav"):
        test_phrase = "Ehinyunyú hinámuhuluka okómítí koko kitwá kiryȃ"

    # Split the phrase into word tokens
    words = tokenize_phrase(test_phrase)

    # Parse the tone pattern from the phrase
    tones = parse_tones(test_phrase)

    # Check that each word has one tone pattern
    is_valid = validate_melodies(words, tones)

    # Print parsing results
    print("\rPhrase and tone parsing tested successfully:")
    print(f"  - Phrase: {test_phrase}")
    print(f"  - Words: {words}")
    print(f"  - Tones: {tones}")
    print(f"  - Valid: {is_valid}")
    print()

    # Make the structured Kinande word sequence
    print("\rMaking word sequence...", end="")
    word_sequence = make_word_sequence(words, tones)

    # Print word sequence
    print("\rWord sequence created successfully:")
    for word_data in word_sequence:
        print(f"  - {word_data}")
    print()

    # Make the structured surrogate note sequence
    print("\rMaking surrogate sequence...", end="")
    surrogate_sequence = make_surrogate_sequence(intervals, surrogate_tones)

    # Print surrogate sequence
    print("\rSurrogate sequence created successfully:")
    print(f"  - Number of surrogate notes: {len(surrogate_sequence)}")
    print(f"  - Surrogate notes:")
    for note_data in surrogate_sequence:
        print(f"    - {note_data}")
    print()

    # Group surrogate tones by Kinande word
    print("\rGrouping surrogate tones by word...", end="")
    grouped_sequence = group_tones_by_word(word_sequence, surrogate_sequence)

    # Print grouped sequence
    print("\rGrouped sequence created successfully:")
    for group in grouped_sequence:
        print(f"  - {group}")
    print()

    # Test sequence alignment
    print("\rTesting sequence alignment...", end="")
    alignment = align_sequences(word_sequence, surrogate_sequence)

    # Print alignment results
    print("\rSequence alignment tested successfully:")
    print(f"  - Number of aligned items: {len(alignment)}")
    print("  - Aligned items:")
    for item in alignment:
        try:
            print(f"    - {item["spoken"]["word"]}: {item["surrogate"]["tone"]}")
        except TypeError:
            print(f"    - {item["spoken"]["word"]}: -")


    print("\nTesting complete.")


if __name__ == "__main__":
    main()