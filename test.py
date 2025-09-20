import sounddevice as sd
import numpy as np

# This function will be called whenever audio data is available
def print_audio_data(indata, frames, time, status):
    print("Callback called! Frames:", frames, "Status:", status)
    # We don't need to do anything with the data, just confirm it's received.

# Try to open an audio stream and listen
try:
    with sd.RawInputStream(samplerate=16000, callback=print_audio_data, blocksize=4096, dtype='int16', channels=1):
        print("Listening for audio data... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)
except Exception as e:
    print("An error occurred:", e)