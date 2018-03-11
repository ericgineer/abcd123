import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import odessa
import pyaudio

plt.close('all')
            
def loadWavData(phrase, frameSize, skipSize, numCoef, numDataSets):
    # Load some training wav files to get MFCC training data
    FILENAME = "audio/test/" + phrase + ".wav" # Name of wav file
    fs, wavData = scipy.io.wavfile.read(FILENAME)    
    Dw = odessa.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
    
    return Dw

def recordAudio(frameSize, skipSize, numCoef):
    duration = 3 # seconds
    CHANNELS = 1
    fs = 16000

    print('Recording start')
    myrec = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS)
    sd.wait()
    print('Recording stop')
    mfcc = odessa.mfcc.getMfcc(np.reshape(myrec, len(myrec)), fs, frameSize, skipSize, numCoef)
    return mfcc


if __name__ == "__main__":
    """ MFCC parameters """

RATE = 16000                        # Sample rate (Hz)    
frameSize = 10                      # Frame size in ms
CHUNK = int(frameSize/1000 * RATE)  # Number of samples to capture in one stream read
FORMAT = pyaudio.paInt16            # Data format. Set to 16 bit integers
CHANNELS = 1                        # Number of channels
RECORD_SECONDS = 10                 # Number of seconds to record
WAVE_OUTPUT_FILENAME = "output.wav" # Name of output wav file

p = pyaudio.PyAudio()

mfcc = odessa.mfcc()

stream = p.open(format=FORMAT,        # Open audio stream for capture
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    

numPrevFrames = 5 # Number of previous frames to append to the speech data    
frames = np.zeros(CHUNK)
prevFrames = np.zeros(numPrevFrames*CHUNK) # Number of previous frames to append
                               # to the beginning of the speech data

print("Say something....")
for i in range(int(RATE / CHUNK * RECORD_SECONDS)): # Read back audio samples
    data = np.fromstring(stream.read(CHUNK), 'Int16') / 32767
    prevFrames[prevFrames.size-CHUNK:prevFrames.size] = data
    for n in range(0,numPrevFrames-1):
        prevFrames[n*CHUNK:(n+1)*CHUNK] = prevFrames[(n+1)*CHUNK:(n+2)*CHUNK]
    isSpeech = mfcc.silenceDetect(data)
    if isSpeech == 1:
        print("Speech detected!")
        for q in range(int(RATE / CHUNK * 3)): # Capture 3 seconds of audio
            if q == 0:
                frames = np.append(prevFrames, data)
            else:
                frames = np.append(frames, data)
            dataByte = stream.read(CHUNK)
            data = np.fromstring(dataByte, 'Int16') / 32767
        break
    
plt.figure()
plt.plot(frames)

plt.figure()
plt.plot(prevFrames)

sd.playrec(frames, RATE, channels=CHANNELS)