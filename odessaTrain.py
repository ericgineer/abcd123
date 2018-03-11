import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st
import odessa2

plt.close('all')
            
def loadWavData(phrase, frameSize, skipSize, numCoef, numDataSets):
    # Load some training wav files to get MFCC training data
    FILENAME = "audio/" + phrase + "/" + phrase + "1.wav" # Name of wav file
    fs, wavData = scipy.io.wavfile.read(FILENAME)    
    mfccVect = odessa2.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
    if numDataSets > 1:
        Dw = np.zeros((mfccVect.shape[0],mfccVect.shape[1],numDataSets))
        Dw[:,:,0] = mfccVect
        for i in range(1,numDataSets+1):
            FILENAME = "audio/" + phrase + "/" + phrase + str(i) + ".wav" # Name of wav file
            print("Reading wave file " + FILENAME)
            fs, wavData = scipy.io.wavfile.read(FILENAME)
            mfccVect = odessa2.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
            Dw[:,:,i-1] = mfccVect
    else:
        Dw = mfccVect
    return Dw

def inithmm(hmmName, numHmmStates, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight):
    # Load training data
    Dw = loadWavData(hmmName, frameSize, skipSize, numCoef, numDataSets)
    if numDataSets > 1:
        mfcc = Dw[:,:,0]
    else:
        mfcc = Dw
    #np.random.seed(0)
    #d1 = np.random.rand(ydim, xdim)
    
    # Initialize the  HMM
    hmm = odessa2.hmm(numHmmStates, leftToRight, numDataSets)
    
    # Train the HMM
    print("Training the ",hmmName," HMM")
    conv = hmm.train(Dw, numIter)
    
    # Save the state transition matrix A to a file
    hmm.saveData(hmmName)
    return conv, hmm, mfcc
        
if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    numDataSets   = 2 
    
    leftToRight = 1 # Force the use of a left to right HMM model
    
    numDataSets   = 20 
    
    numIter = 15 # number of EM algorithm iterations
    
    
    """ Odessa HMM """
    
    OdessaConv, OdessaHmm, OdessaMfcc = inithmm("odessa", 8, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Play Music HMM """
    
    PlayMusicConv, PlayMusicHmm, PlayMusicMfcc = inithmm("PlayMusic", 8, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Stop Music HMM """
    
    StopMusicConv, StopMusicHmm, StopMusicMfcc = inithmm("StopMusic", 9, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Turn Off The Lights HMM """
    
    TurnOffTheLightsConv, TurnOffTheLightsHmm, TurnOffTheLightsMfcc = inithmm("TurnOffTheLights", 9, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Turn On The Lights HMM """
    
    TurnOnTheLightsConv, TurnOnTheLightsHmm, TurnOnTheLightsMfcc = inithmm("TurnOnTheLights", 9, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ What Time Is It HMM """
    
    WhatTimeIsItConv, WhatTimeIsItHmm, WhatTimeIsItMfcc = inithmm("WhatTimeIsIt", 9, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    plt.figure()
    plt.plot(OdessaConv)    
    plt.plot(PlayMusicConv)
    plt.plot(StopMusicConv)
    plt.plot(TurnOffTheLightsConv)
    plt.plot(TurnOnTheLightsConv)
    plt.plot(WhatTimeIsItConv)