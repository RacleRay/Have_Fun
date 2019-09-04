import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment


def graph_spectrogram(wav_file):
    """Calculate and plot spectrogram for a wav audio file.
    The spectrogram tells us how much different frequencies are present in an audio clip at a moment in time.
    A spectrogram is computed by sliding a window over the raw audio signal, and calculates the most active
    frequencies in each window using a Fourier transform.
    """
    rate, data = get_wav_info(wav_file)
    nfft = 200     # Length of each window segment
    fs = 8000      # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


def get_wav_info(wav_file):
    """Load a wav file"""
    rate, data = wavfile.read(wav_file)
    return rate, data


def match_target_amplitude(sound, target_dBFS):
    """Used to standardize volume of audio clip"""
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def load_raw_audio():
    """Load raw audio files for speech synthesis"""
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/" + filename)
            activates.append(activate)

    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/" + filename)
            backgrounds.append(background)

    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/" + filename)
            negatives.append(negative)

    return activates, negatives, backgrounds