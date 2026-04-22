import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

st.title("ECG Visualization Dashboard")

# Load dataset
df = pd.read_csv("data/mitbih_train.csv")

# Select row
row = st.slider("Select ECG sample", 0, len(df)-1, 0)
signal = df.iloc[row, :-1]

# Filter
def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=125):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

filtered = bandpass_filter(signal)

# Peaks
peaks, _ = find_peaks(filtered, distance=30)

# Heart rate
rr_intervals = np.diff(peaks)
heart_rate = 60 / (np.mean(rr_intervals) / 125)

st.subheader(f"Heart Rate: {heart_rate:.2f} BPM")

# Toggle
view = st.radio("Select View", ["Filtered ECG", "Raw ECG"])

# Window slider
start = st.slider("Select window", 0, len(filtered)-300, 0)

# Plot
fig = plt.figure(figsize=(10,4))

if view == "Filtered ECG":
    plt.plot(filtered[start:start+300])
    plt.plot(peaks, filtered[peaks], "x")
    plt.title("Filtered ECG with Peaks")
else:
    plt.plot(signal[start:start+300])
    plt.title("Raw ECG")

plt.xlabel("Time")
plt.ylabel("Amplitude")

st.pyplot(fig)