import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st
import odessa

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
    sd.playrec(myrec, fs, channels=CHANNELS)
    sd.wait()
    mfcc = odessa.mfcc.getMfcc(np.reshape(myrec, len(myrec)), fs, frameSize, skipSize, numCoef)
    return mfcc


if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    leftToRight = 0 # Force the use of a left to right HMM model
    
    numDataSets   = 0 
    
    
    """ Record some audio """
    mfcc = recordAudio(frameSize, skipSize, numCoef)
    
    """ Odessa HMM """
    
    OdessaHmm = odessa.hmm(8, numDataSets)
    OdessaHmm.loadData("odessa")
    
    """ Play Music HMM """
    
    PlayMusicHmm = odessa.hmm(8, numDataSets)
    PlayMusicHmm.loadData("PlayMusic")
    
    """ Stop Music HMM """
    
    StopMusicHmm = odessa.hmm(9, numDataSets)
    StopMusicHmm.loadData("StopMusic")
    
    """ Turn Off The Lights HMM """
    
    TurnOffTheLightsHmm = odessa.hmm(9, numDataSets)
    TurnOffTheLightsHmm.loadData("TurnOffTheLights")
    
    """ Turn On The Lights HMM """
    
    TurnOnTheLightsHmm = odessa.hmm(9, numDataSets)
    TurnOnTheLightsHmm.loadData("TurnOnTheLights")
    
    """ What Time Is It HMM """
    
    WhatTimeIsItHmm = odessa.hmm(9, numDataSets)
    WhatTimeIsItHmm.loadData("WhatTimeIsIt")
    
    probOdessa, llOdessa, alphaOdessa, betaOdessa, Bodessa = OdessaHmm.probEvidence(mfcc)
    probPlayMusic, llPlayMusic, alphaPlayMusic, betaPlayMusic, BplayMusic = PlayMusicHmm.probEvidence(mfcc)
    probStopMusic, llStopMusic, alphaStopMusic, betaStopMusic, BstopMusic = StopMusicHmm.probEvidence(mfcc)
    probTurnOffTheLights, llTurnOffTheLights, betaTurnOffTheLights, alphaTurnOffTheLights, BturnOffTheLights = TurnOffTheLightsHmm.probEvidence(mfcc)
    probTurnOnTheLights, llTurnOnTheLights, alphaTurnOnTheLights, betaTurnOnTheLights, BturnOnTheLights = TurnOnTheLightsHmm.probEvidence(mfcc)
    probWhatTimeIsIt, llWhatTimeIsIt, alphaWhatTimeIsIt, betaWhatTimeIsIt, BwhatTimeIsIt = WhatTimeIsItHmm.probEvidence(mfcc)     
        
    print("")
    
    print("Odessa HMM: ",llOdessa)
    print("Play Music HMM: ",llPlayMusic)
    print("Stop Music HMM: ",llStopMusic)
    print("Turn Off The Lights HMM: ",llTurnOffTheLights)
    print("Turn On The Lights HMM: ",llTurnOnTheLights)
    print("What Time Is It HMM: ",llWhatTimeIsIt)
    
    
    print("")
    
    results = np.array([[llOdessa, llPlayMusic, llStopMusic, llTurnOffTheLights, llTurnOnTheLights, llWhatTimeIsIt]])
    #results = np.array([probOdessa[-2],probPlayMusic[-2],probStopMusic[-2],probTurnOffTheLights[-2],probTurnOnTheLights[-2],probWhatTimeIsIt[-2]])
    idx = np.argmax(results)
    
    if idx == 0:
        print("Odessa")
    elif idx == 1:
        print("Play music")
    elif idx == 2:
        print("Stop music")
    elif idx == 3:
        print("Turn off the lights")
    elif idx == 4:
        print("Turn on the lights")
    elif idx == 5:
        print("What time is it")
    else:
        print("Error")
        
    print("")
    
    Aodessa = OdessaHmm.A
    AplayMusic = PlayMusicHmm.A
    AstopMusic = StopMusicHmm.A
    AturnOffTheLights = TurnOffTheLightsHmm.A
    AturnOnTheLights = TurnOnTheLightsHmm.A
    AwhatTimeIsIt = WhatTimeIsItHmm.A
    
    A = OdessaHmm.A