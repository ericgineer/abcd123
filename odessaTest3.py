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
        
if __name__ == "__main__":
    """ MFCC parameters """
    frameSize = 25 # Length of the frame in milliseconds
    skipSize  = 10 # Time difference in milliseconds between the start of one frame 
                   # and the start of the next frame
    numCoef   = 13 # Number of MFCC coefficients
    
    numDataSets   = 1 
    
    numStates = 10
    
    ydim = 10
    xdim = 5
    
    numDataSets   = 1 
    
    numStates = 5
    
    # Load "Odessa" training data
    #Dw1 = loadWavData("odessa", frameSize, skipSize, numCoef, numDataSets)
    #OdessaMfcc = Dw1[:,:,0]
    np.random.seed(0)
    Dw1 = np.random.rand(xdim, ydim)
    
    # Initialize the "Odessa" HMM
    hmm1 = odessa2.hmm(numStates, Dw1)
    
    # Train the "Odessa" HMM
    #hmm1.train(Dw1)
    
    A = hmm1.A
    mu = hmm1.mu
    C = hmm1.C
    pEvidence, alpha, beta, B, logLikelihood = hmm1.probEvidence(Dw1)
    gamma, xi, logLikelihood = hmm1.em(Dw1)