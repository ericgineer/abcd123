import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import odessa
import pyaudio
import sys
from multiprocessing import Pipe
import os

plt.close('all')

debug = False 

sigIn, sigOut = Pipe() 
            
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

def callback(indata, frames, time, status):
    global buffer
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
        print(sys.stderr)
    sigIn.send(np.reshape(indata, len(indata)))


if __name__ == "__main__":
    """ MFCC parameters """
    mfccFrameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    """ PyAudio parameters """ 

    RATE = 16000                        # Sample rate (Hz)    
    frameSize = 10                      # Frame size in ms
    CHUNK = int(frameSize/1000 * RATE)  # Number of samples to capture in one stream read
    FORMAT = pyaudio.paInt16            # Data format. Set to 16 bit integers
    CHANNELS = 1                        # Number of channels
    RECORD_SECONDS = 10                 # Number of seconds to record
    WAVE_OUTPUT_FILENAME = "output.wav" # Name of output wav file
    
        
    
    numPrevFrames = 83 # Number of previous frames to append to the speech data    
    frames = np.zeros(CHUNK)
    prevFrames = np.zeros(numPrevFrames*CHUNK) # Number of previous frames to append
                                   # to the beginning of the speech data
                                   
    """ Initialize HMMs """
    
    leftToRight = 0 # Force the use of a left to right HMM model
        
    numDataSets   = 0 # Not used, but is an input parameter for the HMM
    
    """ Odessa HMM """
        
    OdessaHmm = odessa.hmm(8, leftToRight, numDataSets)
    OdessaHmm.loadData("odessa")
    
    """ Play Music HMM """
    
    PlayMusicHmm = odessa.hmm(8, leftToRight, numDataSets)
    PlayMusicHmm.loadData("PlayMusic")
    
    """ Stop Music HMM """
    
    StopMusicHmm = odessa.hmm(9, leftToRight, numDataSets)
    StopMusicHmm.loadData("StopMusic")
    
    """ Turn Off The Lights HMM """
    
    TurnOffTheLightsHmm = odessa.hmm(9, leftToRight, numDataSets)
    TurnOffTheLightsHmm.loadData("TurnOffTheLights")
    
    """ Turn On The Lights HMM """
    
    TurnOnTheLightsHmm = odessa.hmm(9, leftToRight, numDataSets)
    TurnOnTheLightsHmm.loadData("TurnOnTheLights")
    
    """ What Time Is It HMM """
    
    WhatTimeIsItHmm = odessa.hmm(9, leftToRight, numDataSets)
    WhatTimeIsItHmm.loadData("WhatTimeIsIt")
    
    heardOdessa = 0  # Set to 1 if the phrase "Odessa" is recognized
    
    speech = odessa.mfcc()
    
    print("Initializing: Please wait.....")
    with sd.InputStream(samplerate=RATE,channels=CHANNELS,callback=callback,blocksize=CHUNK):
        sd.sleep(1000)
        print("Say something....")
        while True: # Read back audio samples in a loop
            prevFrames[prevFrames.size-CHUNK:prevFrames.size] = sigOut.recv()
            for n in range(0,numPrevFrames-1):
                prevFrames[n*CHUNK:(n+1)*CHUNK] = prevFrames[(n+1)*CHUNK:(n+2)*CHUNK]
            isSpeech = speech.silenceDetect(sigOut.recv())
            if isSpeech == 1:
                frames = np.zeros(CHUNK)
                print("Speech detected!")
                for q in range(int(RATE / CHUNK * 3)): # Capture 3 seconds of audio
                    if q == 0:
                        frames = np.append(prevFrames, sigOut.recv())
                    else:
                        frames = np.append(frames, sigOut.recv())
                
                
                frames = frames[0:frames.size-prevFrames.size]
                #sd.playrec(frames, RATE, channels=CHANNELS)
                #sd.wait()
                mfccVect = odessa.mfcc.getMfcc(frames, RATE, mfccFrameSize, skipSize, numCoef)
                probOdessa, llOdessa, alphaOdessa, betaOdessa, Bodessa = OdessaHmm.probEvidence(mfccVect)
                probPlayMusic, llPlayMusic, alphaPlayMusic, betaPlayMusic, BplayMusic = PlayMusicHmm.probEvidence(mfccVect)
                probStopMusic, llStopMusic, alphaStopMusic, betaStopMusic, BstopMusic = StopMusicHmm.probEvidence(mfccVect)
                probTurnOffTheLights, llTurnOffTheLights, betaTurnOffTheLights, alphaTurnOffTheLights, BturnOffTheLights = TurnOffTheLightsHmm.probEvidence(mfccVect)
                probTurnOnTheLights, llTurnOnTheLights, alphaTurnOnTheLights, betaTurnOnTheLights, BturnOnTheLights = TurnOnTheLightsHmm.probEvidence(mfccVect)
                probWhatTimeIsIt, llWhatTimeIsIt, alphaWhatTimeIsIt, betaWhatTimeIsIt, BwhatTimeIsIt = WhatTimeIsItHmm.probEvidence(mfccVect)  
            
            
                if (debug):
                    print("")
            
                    print("Odessa HMM: ",llOdessa)
                    print("Play Music HMM: ",llPlayMusic)
                    print("Stop Music HMM: ",llStopMusic)
                    print("Turn Off The Lights HMM: ",llTurnOffTheLights)
                    print("Turn On The Lights HMM: ",llTurnOnTheLights)
                    print("What Time Is It HMM: ",llWhatTimeIsIt)
                    print("")
                
                results = np.array([[llOdessa, llPlayMusic, llStopMusic, llTurnOffTheLights, llTurnOnTheLights, llWhatTimeIsIt]])
                idx = np.argmax(results)
                
                if (debug):
                    print(idx) # print the index of the detected speech
                
                print("")
                print("")
                print("")
                
                if idx == 0:
                    fs, dataout = scipy.io.wavfile.read("Alarm03.wav")
                    heardOdessa = 1
                    sd.playrec(dataout, fs, channels=CHANNELS)
                    sd.wait()
                    print("Odessa recognized: say a phrase.")
                elif idx == 1 and heardOdessa == 1:
                    heardOdessa = 0
                    print("Play music")
                elif idx == 2 and heardOdessa == 1:
                    heardOdessa = 0
                    print("Stop music")
                elif idx == 3 and heardOdessa == 1:
                    heardOdessa = 0
                    print("Turn off the lights")
                elif idx == 4 and heardOdessa == 1:
                    heardOdessa = 0
                    print("Turn on the lights")
                elif idx == 5 and heardOdessa == 1:
                    heardOdessa = 0
                    print("What time is it")

                print("")
                print("")
                print("")