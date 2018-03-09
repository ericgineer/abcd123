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
    mfcc = Dw # / np.max(np.abs(Dw1))
    #np.random.seed(0)
    #d1 = np.random.rand(ydim, xdim)
    
    # Initialize the  HMM
    hmm = odessa2.hmm(numHmmStates, mfcc, leftToRight)
    
    hmm.loadData(hmmName)
    return hmm, mfcc


if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    numDataSets   = 1 
    
    leftToRight = 0 # Force the use of a left to right HMM model
    
    ydim = 10
    xdim = 100
    
    numDataSets   = 1 
    
    numStates = 5
    
    numIter = 15 # number of EM algorithm iterations
    
    """ Odessa HMM """
    
    OdessaHmm, OdessaMfcc = inithmm("odessa", 10, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Play Music HMM """
    
    PlayMusicHmm, PlayMusicMfcc = inithmm("PlayMusic", 6, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Stop Music HMM """
    
    StopMusicHmm, StopMusicMfcc = inithmm("StopMusic", 6, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Turn Off The Lights HMM """
    
    TurnOffTheLightsHmm, TurnOffTheLightsMfcc = inithmm("TurnOffTheLights", 8, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ Turn On The Lights HMM """
    
    TurnOnTheLightsHmm, TurnOnTheLightsMfcc = inithmm("TurnOnTheLights", 8, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    """ What Time Is It HMM """
    
    WhatTimeIsItHmm, WhatTimeIsItMfcc = inithmm("WhatTimeIsIt", 8, frameSize, skipSize, numCoef, numDataSets, numIter, leftToRight)
    
    #data = OdessaMfcc
    #data = PlayMusicMfcc
    #data = StopMusicMfcc
    #data = TurnOffTheLightsMfcc
    #data = TurnOnTheLightsMfcc
    data = WhatTimeIsItMfcc
    
    probOdessa, llOdessa, alphaOdessa, betaOdessa, Bodessa = OdessaHmm.probEvidence(data)
    probPlayMusic, llPlayMusic, alphaPlayMusic, betaPlayMusic, BplayMusic = PlayMusicHmm.probEvidence(data)
    probStopMusic, llStopMusic, alphaStopMusic, betaStopMusic, BstopMusic = StopMusicHmm.probEvidence(data)
    probTurnOffTheLights, llTurnOffTheLights, betaTurnOffTheLights, alphaTurnOffTheLights, BturnOffTheLights = TurnOffTheLightsHmm.probEvidence(data)
    probTurnOnTheLights, llTurnOnTheLights, alphaTurnOnTheLights, betaTurnOnTheLights, BturnOnTheLights = TurnOnTheLightsHmm.probEvidence(data)
    probWhatTimeIsIt, llWhatTimeIsIt, alphaWhatTimeIsIt, betaWhatTimeIsIt, BwhatTimeIsIt = WhatTimeIsItHmm.probEvidence(data) 
    
    
    
        
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