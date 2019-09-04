import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from utils import *


Tx = 5511    # The number of time steps input to the model from the spectrogram
             # The number of timesteps of the spectrogram will be 5511.
             # (10s sampled at 44100 Hz, divide spectrogram sampling frequency 8000)
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

# the number of pydub module to synthesize audio clip: 10000    (intervals 10/10000 = 0.001 s)
# the number of steps in the output of the GRU model: Ty = 1375 (intervals 10/1375 ≈ 0.0072 s)


# Process: synthesize training data
#    - Overlaying positive/negative words on the background, not Overlap.
#    - Creating the labels at the same time you overlay. 𝑦⟨𝑡⟩ represent whether or not
#      someone has just finished saying "activate.", within a short time-internal after
#      this moment would be label 1.

def get_random_time_segment(segment_ms, synthesize_clips=10000):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    segment_start = np.random.randint(low=0, high=synthesize_clips - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    segment_start, segment_end = segment_time

    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_end >= previous_start and segment_start <= previous_end:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """
    segment_ms = len(audio_clip)

    segment_time = get_random_time_segment(segment_ms)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)
    previous_segments.append(segment_time)

    new_background = background.overlay(audio_clip, position = segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms, Ty=1375):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.

    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """
    segment_end_y = int(segment_end_ms * Ty / 10000.0)  # 转换为Ty intervals
    for i in range(Ty):
        if segment_end_y < i and i < segment_end_y + 5: # 延长5个times step做label
            y[0, i] = 1

    return y


def create_training_example(background, activates, negatives, Ty=1375):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """
    np.random.seed(18)

    background = background - 20  # 减小background
    y = np.zeros((1, Ty))
    previous_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background,
                                                     random_activate,
                                                     previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end, Ty)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, _ = insert_audio_clip(background,
                                          random_negative,
                                          previous_segments)

    # Standardize the volume of the audio clip by add the sub 20
    background = match_target_amplitude(background, -20.0)

    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    x = graph_spectrogram("train.wav")

    return x, y


def preprocess_audio(filename):
    """generate own audio"""
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100, microphone always be 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')