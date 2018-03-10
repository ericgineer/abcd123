import sounddevice as sd
import scipy.io.wavfile
import numpy as np

duration = 3 # seconds
CHANNELS = 1
fs = 16000

print('Recording start')
myrec = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS)
sd.wait()
print('Recording stop')

filename = 'audio/WhatTimeIsIt/WhatTimeIsItTest.wav'

scipy.io.wavfile.write(filename,fs,myrec)

fs, dataout = scipy.io.wavfile.read(filename)

myrecording = sd.playrec(dataout, fs, channels=CHANNELS)