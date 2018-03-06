import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import scipy.stats as st
import odessa

plt.close('all')
            
def loadWavData(phrase, frameSize, skipSize, numCoef, numDataSets):
    # Load some training wav files to get MFCC training data
    FILENAME = "audio/" + phrase + "/" + phrase + "1.wav" # Name of wav file
    fs, wavData = scipy.io.wavfile.read(FILENAME)    
    mfccVect = odessa.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
    Dw = np.zeros((mfccVect.shape[0],mfccVect.shape[1],numDataSets))
    Dw[:,:,0] = mfccVect
    for i in range(1,numDataSets+1):
        FILENAME = "audio/" + phrase + "/" + phrase + str(i) + ".wav" # Name of wav file
        print("Reading wave file " + FILENAME)
        fs, wavData = scipy.io.wavfile.read(FILENAME)
        mfccVect = odessa.mfcc.getMfcc(wavData, fs, frameSize, skipSize, numCoef)
        Dw[:,:,i-1] = mfccVect
    return Dw
        
if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    ydim = 5
    xdim = 5
    
    numDataSets   = 1 
    
    numStates = 10
    
    # Load "Odessa" training data
    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
    #OdessaMfcc = Dw1[:,:,0]
    np.random.seed(0)
    Dw1 = np.random.rand(xdim, ydim)
    
    # Initialize the "Odessa" HMM
    hmm1 = odessa.hmm(numStates, Dw1)
    
    # Train the "Odessa" HMM
    hmm1.train(Dw1)
    
    
    # Load "What time is it" training data
    #Dw2 = loadWavData("WhatTimeIsIt", frameSize, skipSize, numCoef, numDataSets)
    #WhatTimeIsItMfcc = Dw2[:,:,0]
    np.random.seed(1)
    Dw2 = np.random.rand(xdim, ydim)
    
    # Initialize the "What time is it" HMM
    hmm2 = odessa.hmm(numStates, Dw2)
    
    # Train the "What time is it" HMM
    hmm2.train(Dw2)
    
    
    # Load "Play music" training data
    #Dw3 = loadWavData("PlayMusic", frameSize, skipSize, numCoef, numDataSets)
    #PlayMusicMfcc = Dw3[:,:,0]
    np.random.seed(3)
    Dw3 = np.random.rand(xdim, ydim)
    
    # Initialize the "Play music" HMM
    hmm3 = odessa.hmm(numStates, Dw3)
    
    # Train the "Play music" HMM
    hmm3.train(Dw3)
    
    
    """ Test HMMs """
    
    mfccIn = Dw1
    
     # Test with "Odessa"
    probOdessa = hmm1.probEvidence(mfccIn)
    
    # Test with "What time is it"
    probWhatTimeIsIt = hmm2.probEvidence(mfccIn)
    
    # Test with "Play music"
    probPlayMusic = hmm3.probEvidence(mfccIn)
    
    print("p(Odessa | Odessa): ",probOdessa)
    print("p(What time is it | Odessa): ",probWhatTimeIsIt)
    print("p(Play music | Odessa): ",probPlayMusic)
    print("")
    
    likelihoodArray = np.array([probOdessa,probWhatTimeIsIt,probPlayMusic])
    
    plt.figure()
    plt.plot(likelihoodArray,'o')
    plt.title('Odessa HMM')
    
    B = hmm3.stateLikelihood(Dw3)
    logLikelihood1, a1 = hmm3._forward(B)
    b1 = hmm3._backward(B)
    
    gamma1 = hmm3.testGamma
    xi1 = hmm3.testXi
    
    a2, b2, gamma2, xi2, logLikelihood2 = hmm3.recursion(B)
    
    A = hmm3.A
    mu = hmm3.mu
    C = hmm3.C
    cov = hmm3.covs